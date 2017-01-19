import tensorflow as tf
from question_networks import GridPrediction, ProjectionQuestionNet
from basics.layer_helpers import weight_variable, bias_variable, conv2d
import numpy as np
import random
import gym
import gym_minecraft
from time import time
import os


class FCRecurrentAgent:
    def __init__(self, obs_dim, num_actions, pred_type, grid_size = 1, num_proj=1, discount=.99, replay_size=50000,
                 target_freq=10000, max_time=20, start_prob=1., end_prob=.1, anneal_rate=500000,
                 load_path=None, checkpoint_name="AuxQ", scope=None):

        self.prob = start_prob
        self.prob_increment = (start_prob - end_prob)/anneal_rate
        self.time = 0

        experiment_time = str(time())
        dir = os.path.dirname(__file__)
        self.checkpoint_path = os.path.join(dir, "checkpoints", checkpoint_name + experiment_time + ".ckpt")


        self.learning_data = []
        self.lossarr = []
        self.weightarr = []

        self.train_env = gym.make('MinecraftEating1-v0')
        self.test_env = gym.make('MinecraftEating1-v0')
        self.train_env.init()
        self.test_env.init()

        self.replay_size = replay_size
        self.replay_memory = []
        self.graph = tf.Graph()
        self.discount=discount
        self.pred_type = pred_type
        self.num_actions = num_actions


        if pred_type == "Empty":
            pass

        if pred_type == "Grid":
            self.grid_size = grid_size
            self.predictions = GridPrediction(obs_dim=obs_dim, graph=self.graph, num_cuts=20)

        if pred_type == "Projection":
            pass

        if pred_type == "Feature":
            pass

        with self.graph.as_default():
            self.batch_size = tf.placeholder(dtype=tf.int32)
            self.max_time = max_time
            self.masklength = int(self.max_time/2)
            self.input_size = obs_dim

            # Initial convolutional layers #

            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.max_time, self.input_size, self.input_size, 3])
            reshaped_input = tf.reshape(self.input, shape=[-1, self.input_size, self.input_size, 3])

            self.conv1_weight = weight_variable(shape=[4, 4, 3, 32], name='conv1_weight')
            self.conv1_bias = bias_variable(shape=[32], name='conv1_bias')
            conv1_layer = tf.nn.relu(conv2d(reshaped_input, self.conv1_weight, 2) + self.conv1_bias)
            self.conv2_weight = weight_variable(shape=[4, 4, 32, 64], name='conv2_weight')
            self.conv2_bias = bias_variable(shape=[64], name='conv2_bias')
            conv2_layer = tf.nn.relu(conv2d(conv1_layer, self.conv2_weight, 2) + self.conv2_bias)

            conv2_layer_shape = conv2_layer.get_shape().as_list()
            flattened_dim = np.prod(conv2_layer_shape[1:])
            flattened2_layer = tf.reshape(conv2_layer, shape=[-1, flattened_dim])

            self.fc3_weight = weight_variable(shape=[flattened_dim, 256], name='fc3_weight')
            self.fc3_bias = bias_variable(shape=[256], name='fc3_bias')
            fc3_layer = tf.nn.relu(tf.matmul(flattened2_layer, self.fc3_weight) + self.fc3_bias)

            reshaped_fc3_layer = tf.reshape(fc3_layer, shape=[-1, self.max_time, 256])

            # Recurrent connection #

            rec_layer = tf.nn.rnn_cell.BasicRNNCell(256)
            in_state = rec_layer.zero_state(batch_size=self.batch_size, dtype=tf.float32)

            self.rec_output, self.rec_state = tf.nn.dynamic_rnn(cell=rec_layer, inputs=reshaped_fc3_layer,
                                                        dtype=tf.float32, initial_state=in_state, scope=scope)

            reshaped_rec_output = tf.reshape(self.rec_output, shape=[-1, 256])

            # Agent branch (standard Q-value output) #

            self.fc4Q_weight = weight_variable(shape=[256, self.num_actions], name='fc4Q_weight')
            fc4Q_layer = tf.matmul(reshaped_rec_output, self.fc4Q_weight)
            self.Q_output = tf.reshape(fc4Q_layer, shape=[-1, self.max_time, self.num_actions])
            self.Q_final_output = tf.squeeze(tf.slice(self.Q_output, begin=(0, self.max_time - 1, 0),
                                                        size=(-1, 1, -1)), squeeze_dims=[1])

            self.Q_trainable_output = tf.slice(self.Q_output, begin=(0, self.masklength, 0), size=(-1, -1, -1))
            self.Q_target = tf.placeholder(tf.float32, [None, self.max_time - self.masklength])
            self.action_hot = tf.placeholder('float', [None, self.num_actions])
            Q_action_readout = tf.reduce_sum(tf.mul(self.Q_trainable_output, self.action_hot), reduction_indices=[2])
            self.Q_loss = tf.reduce_mean(tf.reduce_sum(.5*tf.square(tf.sub(Q_action_readout, self.Q_target)), reduction_indices=[1]))

            # Prediction branch #

            if pred_type == "Grid":
                self.fc4P_weight = weight_variable(shape=[256, 32*10*10], name='fc4P_weight')
                fc4P_layer = tf.nn.relu(tf.matmul(reshaped_rec_output, self.fc4P_weight))
                self.reshaped_fc4P_layer = tf.reshape(fc4P_layer, shape=[-1, 10, 10, 32])

                self.deconv5P_weight = weight_variable(shape=[4, 4, self.num_actions, 32], name='deconv5P_weight')
                deconv5P_layer = tf.nn.conv2d_transpose(self.reshaped_fc4P_layer, self.deconv5P_weight,
                                                        output_shape=[self.batch_size*self.max_time, self.predictions.num_cuts, self.predictions.num_cuts, self.num_actions],
                                                        strides=[1,2,2,1], padding='SAME')
                self.P_output = tf.reshape(deconv5P_layer, shape=[-1, self.max_time, self.predictions.num_cuts, self.predictions.num_cuts, self.num_actions])

                self.P_trainable_output = tf.slice(self.P_output, begin=(0, self.masklength, 0, 0, 0), size=(-1, -1, -1, -1, -1))
                self.P_target = tf.placeholder(tf.float32, [None, self.max_time - self.masklength, self.predictions.num_cuts, self.predictions.num_cuts])
                P_action_readout = tf.reduce_sum(tf.mul(self.P_trainable_output, self.action_hot), reduction_indices=[4])
                self.P_loss = tf.reduce_mean(tf.reduce_sum(.5*tf.square(tf.sub(P_action_readout, self.P_target)), reduction_indices=[1, 2, 3]))

            # Final loss and gradient calculation. Clipped to global norm 1.

            self.loss = self.Q_loss + self.P_loss

            self.optimizer = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=0.01)
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
            self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
            self.vars = [gravar[1] for gravar in self.gradients_and_vars]
            self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 1.)[0]
            self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.vars))

            initializer = tf.initialize_all_variables()
            self.saver = tf.train.Saver()

        # Load model if given checkpoint path. Otherwise initialize variables.

        self.sess = tf.Session(graph=self.graph)

        if load_path is not None:
            self.saver.restore(self.sess, save_path=load_path)

        else:
            self.sess.run(initializer)

        # Create list of weight variables for use in slow changing target weight network
        self.weights = [self.conv1_weight, self.conv1_bias,
                        self.conv2_weight, self.conv2_bias,
                        self.fc3_weight, self.fc3_bias,
                        # self.graph.get_tensor_by_name("RNN/BasicRNNCell/Linear/Matrix:0"),
                        # self.graph.get_tensor_by_name("RNN/BasicRNNCell/Linear/Bias:0"),
                        self.fc4Q_weight,
                        self.fc4P_weight, self.deconv5P_weight]


    def update_replay_memory(self, tuple):
        # appends tuple and pops old tuple if memory size is exceeded
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)

    def network_step(self, obs, env):
        # helper that steps in environment, updates exploration probability, and properly formats observation

        Q_vals = self.sess.run(self.Q_final_output, feed_dict={self.input: [obs], self.batch_size: 1})
        if random.uniform(0,1) > self.prob:
            step_action = Q_vals.argmax()
        else:
            step_action = random.choice(range(self.num_actions))

        if self.prob > 0.:
            self.prob -= self.prob_increment

        new_obs, step_reward, step_done, _ = env.step(step_action)
        self.time += 1

        return step_action, step_reward, step_done, new_obs, Q_vals.max()

    def test_network(self):
        # test performance and record data into npy files.
        weights = self.sess.run(self.weights)
        layer1_weight_avg = np.average(np.absolute(weights[0]))
        layer1_bias_avg = np.average(np.absolute(weights[1]))
        layer2_weight_avg = np.average(np.absolute(weights[2]))
        layer2_bias_avg = np.average(np.absolute(weights[3]))
        layer3_weight_avg = np.average(np.absolute(weights[4]))
        layer3_bias_avg = np.average(np.absolute(weights[5]))
        layerrec_weight_avg = np.average(np.absolute(weights[6]))
        layerrec_bias_avg = np.average(np.absolute(weights[7]))
        layer4Q_weight_avg = np.average(np.absolute(weights[8]))
        layer4P_weight_avg = np.average(np.absolute(weights[9]))
        layer5P_weight_avg = np.average(np.absolute(weights[10]))

        weight_avgs = [layer1_weight_avg, layer1_bias_avg, layer2_weight_avg, layer2_bias_avg, layer3_weight_avg,
                        layer3_bias_avg, layerrec_weight_avg, layerrec_bias_avg, layer4Q_weight_avg,
                        layer4P_weight_avg, layer5P_weight_avg]

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
                              np.mean(self.lossarr[-100]), self.prob])

        self.weightarr.append(weight_avgs)
        np.save('learning_data', self.learning_data)
        np.save('weight_averages', self.weightarr)
        np.save('weights_' + str(int(self.time/50000)), weights)
        self.saver.save(self.sess, save_path=self.checkpoint_path)
        print("Model checkpoint saved.")

    def learn(self):
        # encapsulates the training procedure

        pass


class DCRecurrentAgent:
    def __init__(self):
        pass


class FCPredictiveStateAgent:
    def __init__(self):
        pass


class DCPredictiveStateAgent:
    def __init__(self):
        pass


#############################

agent = FCRecurrentAgent(obs_dim=80, num_actions=4, pred_type="Grid", start_prob=1., end_prob=.1,
                         anneal_rate=5000, checkpoint_name="Auxpredictor_Grid")
filler = np.ones(shape=[12, 20, 80, 80, 3])
preds = agent.sess.run(agent.P_output, feed_dict={agent.input: filler, agent.batch_size: 12})
print(preds)
print(np.shape(preds))