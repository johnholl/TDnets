from gridworld import IMaze
import tensorflow as tf
from layer_helpers import weight_variable, bias_variable
import random
import numpy as np

class Agent:
    def __init__(self, replay_size=20000):
        self.env = IMaze()
        self.OUTPUT_SIZE = 4
        self.MAX_DEPTH = 10
        self.replay_memory = []
        self.REPLAY_SIZE = replay_size

        # Initialize network
        self.sess = tf.Session()
        self.input = tf.placeholder(tf.float32, shape=[None, 4])
        self.l1weights = weight_variable(shape=[4, 10], name='l1weights')
        self.l1bias = bias_variable(shape=[10], name='l1bias')
        self.layer1 = tf.nn.relu(tf.matmul(self.input, self.l1weights) + self.l1bias)

        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(10, state_is_tuple=True)
        self.rec_input = tf.reshape(self.layer1, shape=[-1, self.MAX_DEPTH, 10])

        self.batch_size = tf.placeholder(tf.int32)
        self.in_state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        self.rec_output, self.rec_state = tf.nn.dynamic_rnn(cell=self.lstm, inputs=self.rec_input,
                                                            dtype=tf.float32, initial_state=self.in_state)

        self.final_rec_output = tf.squeeze(tf.slice(self.rec_output, begin=[0,self.MAX_DEPTH-1,0],
                                                    size=[-1, 1, -1]), squeeze_dims=[1])

        self.l2weights = weight_variable(shape=[10, self.OUTPUT_SIZE], name='l2weights')
        self.l2bias = bias_variable(shape=[self.OUTPUT_SIZE], name='l2bias')

        self.output = tf.matmul(self.final_rec_output, self.l2weights) + self.l2bias

        self.target = tf.placeholder(tf.float32, shape=None)
        self.action_hot = tf.placeholder('float', [None, self.OUTPUT_SIZE])
        self.action_readout = tf.reduce_sum(tf.mul(self.output, self.action_hot), reduction_indices=1)
        self.loss = tf.reduce_mean(.5*tf.square(tf.sub(self.action_readout, self.target)))

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=0.01)
        self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
        self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
        self.vars = [gravar[1] for gravar in self.gradients_and_vars]
        self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 1.)[0]
        self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.vars))

        self.sess.run(tf.initialize_all_variables())

        self.weights = [self.l1weights, self.l1bias, self.l2weights, self.l2bias]

    def update_replay_memory(self, tuple):
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > self.REPLAY_SIZE:
            self.replay_memory.pop(0)


    def true_step(self, prob, obs_sequence, env):
        Q_vals = self.sess.run(self.output, feed_dict={self.input: obs_sequence, self.batch_size: 1})
        if random.uniform(0,1) > prob:
            step_action = Q_vals.argmax()
        else:
            step_action = env.sample_action()

        if prob > 0.1:
            prob -= .000018

        new_obs, step_reward, step_done = env.step(step_action)


        return prob, step_action, step_reward, new_obs, Q_vals.max(), step_done

    def test_network(self):
        # run for 20 episodes. Record step length, avg Q value, max reward, avg reward, loss, weight values for
        # each layer ...
        weights = self.sess.run(self.weights)
        layer1_weight_avg = np.average(np.absolute(weights[0]))
        layer1_bias_avg = np.average(np.absolute(weights[1]))
        layer2_weight_avg = np.average(np.absolute(weights[2]))
        layer2_bias_avg = np.average(np.absolute(weights[3]))
        weight_avgs = [layer1_weight_avg, layer1_bias_avg, layer2_weight_avg, layer2_bias_avg]

        test_env = IMaze()
        total_reward = 0.
        total_steps = 0.
        Q_avg_total = 0.
        max_reward = 0.
        for ep in range(10):
            episode_reward = 0.
            num_steps = 0.
            ep_Q_total = 0.
            obs_sequence = np.zeros(shape=[10, 4])
            obs = test_env.reset()
            obs = np.expand_dims(obs, axis=0)
            done = False
            while not done:
                # test_env.render()
                obs_sequence = np.append(obs_sequence, obs, axis=0)
                obs_sequence = np.delete(obs_sequence, 0, axis=0)
                _, action, reward, new_obs, Qval, done = self.true_step(0.05, obs_sequence, test_env)
                obs = np.expand_dims(new_obs, axis=0)
                episode_reward += reward
                num_steps += 1.
                ep_Q_total += Qval
            max_reward = max(episode_reward, max_reward)
            ep_Q_avg = ep_Q_total/num_steps
            Q_avg_total += ep_Q_avg
            total_reward += episode_reward
            total_steps += num_steps

        avg_Q = Q_avg_total/10.
        avg_reward = total_reward/10.
        avg_steps = total_steps/10.
        print("Average Q-value: {}".format(avg_Q))
        print("Average episode reward: {}".format(avg_reward))
        print("Average number of steps: {}".format(avg_steps))
        print("Max reward over 20 episodes: {}".format(max_reward))

        return weight_avgs, avg_Q, avg_reward, max_reward, avg_steps
