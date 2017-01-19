import numpy as np
import tensorflow as tf
from chainmrp.chain_question_net import ChainQuestionNet, DiscountQuestionNet
from layer_helpers import weight_variable, bias_variable

from basics.customrnn_cell import MultilayerRNNCell


class ChainAnswerNet:
    def __init__(self, obs_dim, max_time=6, depth=4, load_path=None, replay_size=3000, scope=None, softmax=False):

        self.replay_size = replay_size
        self.replay_memory = []
        self.graph = tf.Graph()
        self.depth = depth
        self.questionnet = ChainQuestionNet(obs_dim=obs_dim, depth=self.depth, graph=self.graph)
        with self.graph.as_default():
            self.batch_size = tf.placeholder(dtype=tf.int32)
            self.max_time = max_time
            self.masklength = int(self.max_time/2)
            self.input_size = self.questionnet.obs_dim
            self.output_size = self.questionnet.output_size

            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.max_time, self.input_size])
            self.rec_layer = tf.nn.rnn_cell.BasicRNNCell(self.output_size)
            self.in_state = self.rec_layer.zero_state(batch_size=self.batch_size, dtype=tf.float32)

            self.rec_output, self.rec_state = tf.nn.dynamic_rnn(cell=self.rec_layer, inputs=self.input,
                                                                dtype=tf.float32, initial_state=self.in_state, scope=scope)


            # rec_output has size [batch size, max time, 24]

            if softmax is False:
                self.final_prediction = tf.squeeze(tf.slice(self.rec_output, begin=(0, self.max_time - 1, 0),
                                                            size=(-1, 1, -1)), squeeze_dims=[1])

                self.trainable_pred_output = tf.slice(self.rec_output, begin=(0, int(self.max_time / 2), 0), size=(-1, -1, -1))
                self.pred_targets = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.max_time - self.masklength, self.output_size])
                self.prediction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.trainable_pred_output - self.pred_targets), reduction_indices=[1, 2]))

                self.loss = self.prediction_loss

            elif softmax is True:
                self.normalized_rec_output = tf.reshape(tf.nn.softmax(
                        tf.reshape(self.rec_output, shape=[-1, self.questionnet.obs_dim])),
                    shape=[-1, self.max_time, self.output_size])
                self.trainable_pred_output = tf.slice(self.rec_output, begin=(0, self.masklength, 0), size=(-1, -1, -1))
                self.reshaped_trainable_output = tf.reshape(self.trainable_pred_output, shape=[-1, self.questionnet.obs_dim])
                self.pred_targets = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.masklength, self.output_size])
                self.reshaped_pred_targets = tf.reshape(self.pred_targets, shape=[-1, self.questionnet.obs_dim])
                self.entropies = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(self.reshaped_trainable_output, self.reshaped_pred_targets),
                                            shape=[self.batch_size, self.masklength, self.depth])
                self.loss = tf.reduce_mean(tf.reduce_sum(self.entropies, reduction_indices=[1, 2]))

            self.optimizer = tf.train.RMSPropOptimizer(0.0025, decay=0.95, epsilon=0.01)

            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
            self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
            self.vars = [gravar[1] for gravar in self.gradients_and_vars]
            self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 1.)[0]
            self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.vars))

            initializer = tf.initialize_all_variables()
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        if load_path is not None:
            self.saver.restore(self.sess, save_path=load_path)

        else:
            self.sess.run(initializer)

        self.weights = [self.graph.get_tensor_by_name("RNN/BasicRNNCell/Linear/Matrix:0"),
                        self.graph.get_tensor_by_name("RNN/BasicRNNCell/Linear/Bias:0")]


    def prediction_from_obs(self, obs):
        # from a single observation and prediction produce the next prediction
        z = np.zeros(shape=[self.max_time-1, self.input_size])
        full_input = [np.concatenate((z, [obs]), axis=0)]
        prediction = self.sess.run(self.rec_output, feed_dict={self.input: full_input})
        return prediction

    def prediction_from_sequence(self, obs_sequence):
        # obs_sequence should be shaped [max depth, obs size]
        full_input = [obs_sequence]
        prediction = self.sess.run(self.rec_output, feed_dict={self.input: full_input})
        return prediction

    def save_data(self, path):
        self.saver.save(self.sess, save_path=path)
        print("Model checkpoint saved.")

    def update_replay_memory(self, example):
        self.replay_memory.append(example)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)


class DiscountChainAnswerNet:

    def __init__(self, obs_dim, max_time=6, load_path=None, replay_size=3000, scope=None, discount=.99):

            self.replay_size = replay_size
            self.replay_memory = []
            self.graph = tf.Graph()
            self.discount=discount
            self.questionnet = DiscountQuestionNet(obs_dim=obs_dim, graph=self.graph, gamma=discount)
            with self.graph.as_default():
                self.batch_size = tf.placeholder(dtype=tf.int32)
                self.max_time = max_time
                self.masklength = int(self.max_time/2)
                self.input_size = self.questionnet.obs_dim
                self.output_size = self.questionnet.output_size

                self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.max_time, self.input_size])
                self.rec_layer = tf.nn.rnn_cell.BasicRNNCell(self.output_size)
                self.in_state = self.rec_layer.zero_state(batch_size=self.batch_size, dtype=tf.float32)

                self.rec_output, self.rec_state = tf.nn.dynamic_rnn(cell=self.rec_layer, inputs=self.input,
                                                                    dtype=tf.float32, initial_state=self.in_state, scope=scope)



                self.final_prediction = tf.squeeze(tf.slice(self.rec_output, begin=(0, self.max_time - 1, 0),
                                                            size=(-1, 1, -1)), squeeze_dims=[1])

                self.trainable_pred_output = tf.slice(self.rec_output, begin=(0, int(self.max_time / 2), 0), size=(-1, -1, -1))
                self.pred_targets = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.max_time - self.masklength, self.output_size])
                self.prediction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.trainable_pred_output - self.pred_targets), reduction_indices=[1, 2]))

                self.loss = self.prediction_loss

                self.optimizer = tf.train.RMSPropOptimizer(0.000025, decay=0.95, epsilon=0.01)

                #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
                self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
                self.vars = [gravar[1] for gravar in self.gradients_and_vars]
                self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 1.)[0]
                self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.vars))

                initializer = tf.initialize_all_variables()
                self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=self.graph)

            if load_path is not None:
                self.saver.restore(self.sess, save_path=load_path)

            else:
                self.sess.run(initializer)

            self.weights = [self.graph.get_tensor_by_name("RNN/BasicRNNCell/Linear/Matrix:0"),
                             self.graph.get_tensor_by_name("RNN/BasicRNNCell/Linear/Bias:0")]




    def prediction_from_obs(self, obs):
        # from a single observation and prediction produce the next prediction
        z = np.zeros(shape=[self.max_time-1, self.input_size])
        full_input = [np.concatenate((z, [obs]), axis=0)]
        prediction = self.sess.run(self.rec_output, feed_dict={self.input: full_input})
        return prediction

    def prediction_from_sequence(self, obs_sequence):
        # obs_sequence should be shaped [max depth, obs size]
        full_input = [obs_sequence]
        prediction = self.sess.run(self.rec_output, feed_dict={self.input: full_input})
        return prediction

    def save_data(self, path):
        self.saver.save(self.sess, save_path=path)
        print("Model checkpoint saved.")

    def update_replay_memory(self, example):
        self.replay_memory.append(example)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)





class BasicDiscountChainAnswerNet:

    def __init__(self, obs_dim, max_time=6, load_path=None, replay_size=3000, scope=None, discount=.99):

            self.replay_size = replay_size
            self.replay_memory = []
            self.graph = tf.Graph()
            self.questionnet = DiscountQuestionNet(obs_dim=1, graph=self.graph, gamma=discount)
            with self.graph.as_default():
                self.batch_size = tf.placeholder(dtype=tf.int32)
                self.max_time = max_time
                self.masklength = int(np.ceil(self.max_time/2.))
                self.input_size = obs_dim
                self.output_size = self.questionnet.output_size

                self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.max_time, self.input_size])
                # self.reshaped_input = tf.reshape(self.input, shape=[-1, self.input_size])
                # self.l1weights = weight_variable(shape=[self.input_size, 50], name='l1weight')
                # self.l1bias = bias_variable(shape=[50], name='l1bias')
                # self.layer1 = tf.nn.sigmoid(tf.matmul(self.reshaped_input, self.l1weights) + self.l1bias)
                # self.reshaped_layer1 = tf.reshape(self.layer1, shape=[-1, self.max_time, 50])
                self.rec_layer = tf.nn.rnn_cell.BasicRNNCell(self.output_size)
                self.in_state = self.rec_layer.zero_state(batch_size=self.batch_size, dtype=tf.float32)

                self.rec_output, self.rec_state = tf.nn.dynamic_rnn(cell=self.rec_layer, inputs=self.input,
                                                                    dtype=tf.float32, initial_state=self.in_state, scope=scope)



                self.final_prediction = tf.squeeze(tf.slice(self.rec_output, begin=(0, self.max_time - 1, 0),
                                                            size=(-1, 1, -1)), squeeze_dims=[1])

                self.trainable_pred_output = tf.slice(self.rec_output, begin=(0, int(self.max_time / 2), 0), size=(-1, -1, -1))
                self.pred_targets = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.max_time - self.masklength, self.output_size])
                self.prediction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.trainable_pred_output - self.pred_targets), reduction_indices=[1, 2]))

                self.loss = self.prediction_loss

                self.optimizer = tf.train.RMSPropOptimizer(0.0025, decay=0.95, epsilon=0.01)

                # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
                self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
                self.vars = [gravar[1] for gravar in self.gradients_and_vars]
                self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 1.)[0]
                self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.vars))

                initializer = tf.initialize_all_variables()
                trainable_variables = tf.trainable_variables()
                self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=self.graph)

            if load_path is not None:
                self.saver.restore(self.sess, save_path=load_path)

            else:
                self.sess.run(initializer)


            self.weights = [self.graph.get_tensor_by_name("RNN/BasicRNNCell/Linear/Matrix:0"),
                            self.graph.get_tensor_by_name("RNN/BasicRNNCell/Linear/Bias:0")]



    def prediction_from_obs(self, obs):
        # from a single observation and prediction produce the next prediction
        z = np.zeros(shape=[self.max_time-1, self.input_size])
        full_input = [np.concatenate((z, [obs]), axis=0)]
        prediction = self.sess.run(self.rec_output, feed_dict={self.input: full_input})
        return prediction

    def prediction_from_sequence(self, obs_sequence):
        # obs_sequence should be shaped [max depth, obs size]
        full_input = [obs_sequence]
        prediction = self.sess.run(self.rec_output, feed_dict={self.input: full_input})
        return prediction

    def save_data(self, path):
        self.saver.save(self.sess, save_path=path)
        print("Model checkpoint saved.")

    def update_replay_memory(self, example):
        self.replay_memory.append(example)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)


class RecurrentAgent:

    def __init__(self, obs_dim, max_time=6, load_path=None, replay_size=3000, scope=None, discount=.99):

            self.replay_size = replay_size
            self.replay_memory = []
            self.graph = tf.Graph()
            self.discount=discount
            self.questionnet = DiscountQuestionNet(obs_dim=obs_dim, graph=self.graph, gamma=discount)
            with self.graph.as_default():
                self.batch_size = tf.placeholder(dtype=tf.int32)
                self.max_time = max_time
                self.masklength = int(self.max_time/2)
                self.input_size = self.questionnet.obs_dim
                self.output_size = self.questionnet.output_size

                self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.max_time, self.input_size])
                self.rec_layer = tf.nn.rnn_cell.BasicRNNCell(self.output_size)
                self.in_state = self.rec_layer.zero_state(batch_size=self.batch_size, dtype=tf.float32)

                self.rec_output, self.rec_state = tf.nn.dynamic_rnn(cell=self.rec_layer, inputs=self.input,
                                                                    dtype=tf.float32, initial_state=self.in_state, scope=scope)



                self.reshaped_rec_output = tf.reshape(self.rec_output, shape=[-1, self.output_size])
                fc_weight = weight_variable(shape=[self.output_size, self.output_size], name='fc_weight')
                fc_bias = bias_variable(shape=[self.output_size], name='fc_bias')
                self.output = tf.add(tf.matmul(self.reshaped_rec_output, fc_weight), fc_bias)
                self.reshaped_output = tf.reshape(self.output, shape=[-1, self.max_time, self.output_size])
                self.final_prediction = tf.squeeze(tf.slice(self.reshaped_output, begin=(0, self.max_time - 1, 0),
                                                            size=(-1, 1, -1)), squeeze_dims=[1])
                self.trainable_pred_output = tf.slice(self.reshaped_output, begin=(0, self.masklength, 0),
                                                      size=(-1, -1, -1))
                self.pred_targets = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.max_time - self.masklength, self.output_size])
                self.prediction_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(self.trainable_pred_output - self.pred_targets), reduction_indices=[1, 2]))

                self.loss = self.prediction_loss

                self.optimizer = tf.train.RMSPropOptimizer(0.0025, decay=0.95, epsilon=0.01)

                #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
                self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
                self.vars = [gravar[1] for gravar in self.gradients_and_vars]
                self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 1.)[0]
                self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.vars))

                initializer = tf.initialize_all_variables()
                self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=self.graph)

            if load_path is not None:
                self.saver.restore(self.sess, save_path=load_path)

            else:
                self.sess.run(initializer)

            self.weights = [self.graph.get_tensor_by_name("RNN/BasicRNNCell/Linear/Matrix:0"),
                            self.graph.get_tensor_by_name("RNN/BasicRNNCell/Linear/Bias:0")]




    def prediction_from_obs(self, obs):
        # from a single observation and prediction produce the next prediction
        z = np.zeros(shape=[self.max_time-1, self.input_size])
        full_input = [np.concatenate((z, [obs]), axis=0)]
        prediction = self.sess.run(self.rec_output, feed_dict={self.input: full_input})
        return prediction

    def prediction_from_sequence(self, obs_sequence):
        # obs_sequence should be shaped [max depth, obs size]
        full_input = [obs_sequence]
        prediction = self.sess.run(self.rec_output, feed_dict={self.input: full_input})
        return prediction

    def save_data(self, path):
        self.saver.save(self.sess, save_path=path)
        print("Model checkpoint saved.")

    def update_replay_memory(self, example):
        self.replay_memory.append(example)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)


class MultilayerRecurrentAgent:


    def __init__(self, obs_dim, max_time=6, load_path=None, replay_size=3000, scope=None, discount=.99):

            self.replay_size = replay_size
            self.replay_memory = []
            self.graph = tf.Graph()
            self.discount=discount
            self.questionnet = DiscountQuestionNet(obs_dim=obs_dim, graph=self.graph, gamma=discount)
            with self.graph.as_default():
                self.batch_size = tf.placeholder(dtype=tf.int32)
                self.max_time = max_time
                self.masklength = int(self.max_time/2)
                self.input_size = self.questionnet.obs_dim
                self.output_size = self.questionnet.output_size

                self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.max_time, self.input_size])
                self.rec_layer = MultilayerRNNCell(self.output_size)
                self.in_state = self.rec_layer.zero_state(batch_size=self.batch_size, dtype=tf.float32)

                self.rec_output, self.rec_state = tf.nn.dynamic_rnn(cell=self.rec_layer, inputs=self.input,
                                                                    dtype=tf.float32, initial_state=self.in_state, scope=scope)



                self.final_prediction = tf.squeeze(tf.slice(self.rec_output, begin=(0, self.max_time - 1, 0),
                                                            size=(-1, 1, -1)), squeeze_dims=[1])

                self.trainable_pred_output = tf.slice(self.rec_output, begin=(0, int(self.max_time / 2), 0), size=(-1, -1, -1))
                self.pred_targets = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.max_time - self.masklength, self.output_size])
                self.prediction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.trainable_pred_output - self.pred_targets), reduction_indices=[1, 2]))

                self.loss = self.prediction_loss

                self.optimizer = tf.train.RMSPropOptimizer(0.0025, decay=0.95, epsilon=0.01)

                #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
                self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
                self.vars = [gravar[1] for gravar in self.gradients_and_vars]
                self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 1.)[0]
                self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.vars))

                initializer = tf.initialize_all_variables()
                self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=self.graph)

            if load_path is not None:
                self.saver.restore(self.sess, save_path=load_path)

            else:
                self.sess.run(initializer)

            self.weights = [self.graph.get_tensor_by_name("RNN/MultilayerRNNCell/firstlayer/Matrix:0"),
                            self.graph.get_tensor_by_name("RNN/MultilayerRNNCell/firstlayer/Bias:0"),
                            self.graph.get_tensor_by_name("RNN/MultilayerRNNCell/secondlayer/Matrix:0"),
                            self.graph.get_tensor_by_name("RNN/MultilayerRNNCell/secondlayer/Bias:0")]

    def prediction_from_obs(self, obs):
        # from a single observation and prediction produce the next prediction
        z = np.zeros(shape=[self.max_time-1, self.input_size])
        full_input = [np.concatenate((z, [obs]), axis=0)]
        prediction = self.sess.run(self.rec_output, feed_dict={self.input: full_input})
        return prediction

    def prediction_from_sequence(self, obs_sequence):
        # obs_sequence should be shaped [max depth, obs size]
        full_input = [obs_sequence]
        prediction = self.sess.run(self.rec_output, feed_dict={self.input: full_input})
        return prediction

    def save_data(self, path):
        self.saver.save(self.sess, save_path=path)
        print("Model checkpoint saved.")

    def update_replay_memory(self, example):
        self.replay_memory.append(example)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)


