from __future__ import print_function
import distutils.version
import scipy.signal
import threading
from collections import namedtuple
import numpy as np
import six.moves.queue as queue
import tensorflow as tf
from model import AuxLSTMPolicy
import random
import sys

use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_pa = np.asarray(rollout.prev_action)
    batch_pr = np.asarray(rollout.prev_reward)
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features, batch_pa, batch_pr)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features", "pa", "pr"])

class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.prev_action = []
        self.prev_reward = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features, prev_action, prev_reward):
        self.prev_action += [prev_action]
        self.prev_reward += [prev_reward]
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.prev_action.extend(other.prev_action)
        self.prev_reward.extend(other.prev_reward)
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)

class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
    def __init__(self, env, policy, num_local_steps):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)




def env_runner(env, policy, num_local_steps, summary_writer):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    last_state = env.reset()
    last_features = policy.get_initial_features()
    length = 0
    rewards = 0
    prev_reward = 0
    prev_action = np.zeros(shape=[env.num_actions])

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            action_index, value_, features = policy.act(last_state, prev_action, prev_reward, *last_features)
            action = np.array([float(int(i == action_index)) for i in range(env.num_actions)])

            state, reward, terminal = env.step(action_index)

            # collect the experience
            rollout.add(last_state, action, reward, value_, terminal, last_features, prev_action, [prev_reward])
            policy.update_replay_memory((last_state, action, reward, terminal,
                                         last_features, prev_action, [prev_reward],
                                         state, value_
                                         ))
            length += 1
            rewards += reward

            prev_action = action
            prev_reward = reward
            last_state = state
            last_features = features

            if terminal:
                summary = tf.Summary()
                summary.value.add(tag='episode_reward', simple_value=float(rewards))
                summary.value.add(tag='reward_per_timestep', simple_value=float(rewards)/float(length))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()
                terminal_end = True
                # if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                last_state = env.reset()
                last_features = policy.get_initial_features()
                print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
                length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, prev_action, prev_reward, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout

class A3C(object):
    def __init__(self, env, task):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""

        self.env = env
        self.task = task
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = AuxLSTMPolicy(env.observation_space_shape, env.num_actions)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = AuxLSTMPolicy(env.observation_space_shape, env.num_actions)
                pi.global_step = self.global_step

                self.Q_target = tf.placeholder(tf.float32, [None])
                self.action = tf.placeholder(tf.float32, [None, env.num_actions])


                if use_tf12_api:
                    Q_readout = tf.reduce_sum(tf.multiply(pi.Q, self.action), reduction_indices=[1])
                else:
                    Q_readout = tf.reduce_sum(tf.mul(pi.Q, self.action), reduction_indices=[1])

                self.loss = tf.reduce_sum(tf.square(Q_readout - self.Q_target))

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, 20)
            bs = tf.to_float(tf.shape(pi.x)[0])

            grads = tf.gradients(self.loss, pi.var_list)

            if use_tf12_api:
                tf.summary.scalar("model/loss", self.loss / bs)
                tf.summary.image("model/state", pi.x*255, max_outputs=12)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                for var in pi.var_list:
                    tf.summary.scalar("model/" + var.name + "_norm", tf.global_norm([var]))
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/policy_loss", self.loss / bs)
                tf.image_summary("model/state", pi.x*255, max_images=12)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                for var in pi.var_list:
                    tf.scalar_summary("model/" + var.name + "_norm", tf.global_norm([var]))
                self.summary_op = tf.merge_all_summaries()

            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            # thread_lr = 10.**(np.random.uniform(-4, -2.30103))
            thread_lr = .0001
            opt = tf.train.RMSPropOptimizer(thread_lr)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""

        sess.run(self.sync)  # copy weights from shared to local

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        rollout = self.pull_batch_from_queue()

        """
        Grab a minibatch from the replay memory. Set targets and update parameters
        from the prediction loss function"""
        if len(self.runner.policy.replay_memory) > 100:
            Qbatch = []
            starting_pos = random.choice(range(len(self.runner.policy.replay_memory)-20))
            terminal = False
            batchsize = 0
            while not terminal and batchsize < 20:
                Qbatch.append(self.runner.policy.replay_memory[starting_pos + batchsize])
                batchsize += 1
                terminal = Qbatch[-1][3]

            last_states = [m[0] for m in Qbatch]
            last_actions = [m[5] for m in Qbatch]
            last_rewards = [m[6] for m in Qbatch]
            rewards = [m[2] for m in Qbatch]
            actions = [m[1] for m in Qbatch]
            start_features = Qbatch[0][4]
            states = [m[7] for m in Qbatch]

            feed_dict={self.local_network.x: states,
                       self.local_network.action: actions,
                       self.local_network.reward: np.expand_dims(rewards, axis=1),
                       self.local_network.state_in[0]: start_features[0],
                       self.local_network.state_in[1]: start_features[1]}

            feed_dict.update(zip(self.local_network.var_list, self.local_network.target_weights))


            Q_values, values = sess.run((self.local_network.Q, self.local_network.vf), feed_dict=feed_dict)


            Q_targets = []
            for i in range(len(Qbatch)):
                new_Q_target = 0.
                for j in range(len(Qbatch)-i):
                    new_Q_target += (.99**j)*rewards[i+j]
                Q_targets.append(new_Q_target)
                if not Qbatch[-1][3]:
                    Q_targets[-1] += (.99**(len(Qbatch)-i))*values[-1]

            # put fetches logic here
            if should_compute_summary:
                fetches = [self.summary_op, self.train_op, self.global_step]
            else:
                fetches = [self.train_op, self.global_step]

            fetched = sess.run(fetches, feed_dict={self.local_network.x: last_states,
                                                          self.local_network.action: last_actions,
                                                          self.local_network.reward: last_rewards,
                                                          self.local_network.state_in[0]: start_features[0],
                                                          self.local_network.state_in[1]: start_features[1],
                                                          self.Q_target: Q_targets,
                                                          self.action: actions})




            if should_compute_summary:
                self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
                self.summary_writer.flush()
        self.local_steps += 1