import os
import random
from time import time

import gym
import numpy as np
import tensorflow as tf

from basics.layer_helpers import weight_variable, bias_variable


class TaxiQNet:

    # A Q network designed to learn OpenAI's 5x5 taxi environment

    def __init__(self, conv=False, load_path=None, replay_size=1000,
                 prob= 1., anneal_rate=5000., tgt_update_freq=3000,
                 checkpoint_name="TaxiQNet"):
        self.train_env = gym.make('Taxi-v1')
        self.test_env = gym.make('Taxi-v1')

        self.output_size = 6
        self.replay_memory = []
        self.replay_size = replay_size
        self.prob = prob
        self.prob_increment = 1./anneal_rate
        self.conv = conv
        self.tgt_update_freq = tgt_update_freq
        self.time = 0
        self.learning_data = []
        self.loss_arr = []

        experiment_time = str(time())
        dir = os.path.dirname(__file__)
        self.checkpoint_path = os.path.join(dir, "checkpoints", checkpoint_name + experiment_time + ".ckpt")

        self.graph = tf.Graph()

        if conv==False:
            self.weights, self.input, self.output, self.target, self.action_hot, self.loss = self._initialize_standard()

        elif conv==True:
            # self.weights, self.input, self.output, self.target, self.action_hot, self.loss = self._initialize_convolutional()
            pass

        with self.graph.as_default():
            optimizer = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=0.01)
            gradients_and_vars = optimizer.compute_gradients(self.loss)
            gradients = [gravar[0] for gravar in gradients_and_vars]
            vars = [gravar[1] for gravar in gradients_and_vars]
            clipped_gradients = tf.clip_by_global_norm(gradients, 1.)[0]
            self.train_operation = optimizer.apply_gradients(zip(clipped_gradients, vars))

            initializer = tf.initialize_all_variables()
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=self.graph)

        if load_path is not None:
            self.saver.restore(self.sess, save_path=load_path)
        else:
            self.sess.run(initializer)

    def _initialize_convolutional(self):
        # create a convolutional network (used when observations are formatted as 2d with several filters)
        pass

    def _initialize_standard(self):
        # creates a network where inputs are one-hot encodings of observation
        with self.graph.as_default():
            obs_batch = tf.placeholder(tf.float32, shape=[None, 500])
            weight1 = weight_variable(shape=[500, 100], name='l1w')
            bias1 = bias_variable(shape=[100], name='l1b')
            layer1 = tf.nn.relu(tf.matmul(obs_batch, weight1) + bias1)
            weight2 = weight_variable(shape=[100,30], name='l2w')
            bias2 = bias_variable(shape=[30], name='l2b')
            layer2 = tf.nn.relu(tf.matmul(layer1, weight2) + bias2)
            weight3 = weight_variable(shape=[30, self.output_size], name='l3w')
            output = tf.matmul(layer2, weight3)
            target = tf.placeholder(tf.float32, None)
            action_hot = tf.placeholder('float', [None, self.output_size])
            action_readout = tf.reduce_sum(tf.mul(output, action_hot), reduction_indices=1)
            loss = tf.reduce_mean(.5*tf.square(tf.sub(action_readout, target)))

        weights = [weight1, bias1, weight2, bias2, weight3]

        return weights, obs_batch, output, target, action_hot, loss

    def update_replay_memory(self, tuple):
        # appends tuple and pops old tuple if memory size is exceeded
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)

    def network_step(self, obs, env):
        # helper that steps in environment, updates exploration probability, and properly formats observation
        if self.conv:
            formatted_obs = self.format_to_conv(obs)
        else:
            formatted_obs = self.format_to_standard(obs)

        Q_vals = self.sess.run(self.output, feed_dict={self.input: [formatted_obs]})
        if random.uniform(0,1) > self.prob:
            step_action = Q_vals.argmax()
        else:
            step_action = random.choice([0,1,2,3,4,5])

        if self.prob > 0.:
            self.prob -= self.prob_increment

        new_obs, step_reward, step_done, _ = env.step(step_action)
        self.time += 1

        return step_action, step_reward, step_done, new_obs, Q_vals.max()

    def format_to_conv(self, obs):
        # converts integer observation into 2d image with filters for use in convolutional network
        pass

    def format_to_standard(self, obs):
        # converts integer observation into one-hot vector for use in standard network
        obs_one_hot = np.zeros(shape=[500])
        obs_one_hot[obs] = 1.
        return obs_one_hot

    def test_network(self):
        # test performance and record data into npy files.
        weights = self.sess.run(self.weights)
        layer1_weight_avg = np.average(np.absolute(weights[0]))
        layer1_bias_avg = np.average(np.absolute(weights[1]))
        layer2_weight_avg = np.average(np.absolute(weights[2]))
        layer2_bias_avg = np.average(np.absolute(weights[3]))
        layer3_weight_avg = np.average(np.absolute(weights[4]))
        layer3_bias_avg = np.average(np.absolute(weights[5]))
        layer4_weight_avg = np.average(np.absolute(weights[6]))
        layer4_bias_avg = np.average(np.absolute(weights[7]))
        weight_avgs = [layer1_weight_avg, layer1_bias_avg, layer2_weight_avg, layer2_bias_avg, layer3_weight_avg,
                         layer3_bias_avg, layer4_weight_avg, layer4_bias_avg]

        total_reward = 0.
        total_steps = 0.
        Q_avg_total = 0.
        max_reward = 0.
        for ep in range(10):
            obs = self.test_env.reset()
            episode_reward = 0.
            num_steps = 0.
            ep_Q_total = 0.
            done = False
            while not done:
                action, reward, done, new_obs, Qval = self.network_step(obs=obs, env=self.test_env)
                obs = new_obs
                episode_reward += reward
                num_steps += 1.
                ep_Q_total += Qval
            max_reward = max(episode_reward, max_reward)
            ep_Q_avg = ep_Q_total/num_steps
            Q_avg_total += ep_Q_avg
            total_reward += episode_reward
            total_steps += num_steps

        avg_Q = Q_avg_total/20.
        avg_reward = total_reward/20.
        avg_steps = total_steps/20.
        print("Average Q-value: {}".format(avg_Q))
        print("Average episode reward: {}".format(avg_reward))
        print("Average number of steps: {}".format(avg_steps))
        print("Max reward over 20 episodes: {}".format(max_reward))

        return weight_avgs, avg_Q, avg_reward, max_reward, avg_steps, weights

    def save_data(self):
        weight_avgs, avg_Q, avg_rewards, max_reward, avg_steps, weights = self.test_network()
        self.learning_data.append([self.time, avg_Q, avg_rewards, max_reward, avg_steps,
                              np.mean(self.loss_arr[-100]), self.prob])

        self.weightarr.append(weight_avgs)
        np.save('learning_data', self.learning_data)
        np.save('weight_averages', self.weightarr)
        np.save('weights_' + str(int(self.time/50000)), weights)
        self.saver.save(self.sess, save_path=self.checkpoint_path)
        print("Model checkpoint saved.")

    def learn(self):
        # encapsulates the training procedure

        pass




# class TaxiQQuestionNet:
#     def __init__(self):
#         pass



