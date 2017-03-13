import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from layer_helpers import weight_variable, bias_variable

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(flatten(x), [0])

        size = 256
        lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]



class AuxLSTMPolicy(object):
    def __init__(self, ob_space, ac_space, replay_size=2000):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])

        self.replay_size = replay_size
        self.replay_memory = []
        self.prob = 1.
        self.final_prob = 0.1
        self.anneal_rate = .00000018
        self.num_actions = ac_space

        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
        x = conv_features = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
        x = flatten(x)
        x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
        x = tf.concat(concat_dim=1, values=[x, self.action, self.reward])
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(x, [0])

        size = 256
        lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state

        x = tf.reshape(lstm_outputs, [-1, size])
        self.Q = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reduce_max(self.Q, axis=[1])
        # self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # A way to sample actions weighted by Q-values. Probably not a good way to explore
        self.sample = categorical_sample(self.Q, ac_space)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.target_weights = []


    def get_initial_features(self):
        return self.state_init


    def act(self, ob, prev_a, prev_r, c, h):
        sess = tf.get_default_session()
        self.update_exploration()
        fetched = sess.run([self.Q, self.vf] + self.state_out,
                                {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                 self.state_in[0]: c, self.state_in[1]: h})
        qvals = fetched[0]

        if np.random.uniform > self.prob:
            action = np.argmax(qvals)
        else:
            action = np.random.choice(range(self.num_actions))

        return action, fetched[1], fetched[2:]



    def value(self, ob, prev_a, prev_r, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                  self.state_in[0]: c, self.state_in[1]: h})[0]

    def update_replay_memory(self, tuple):
        # appends tuple and pops old tuple if memory size is exceeded
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)

    def update_exploration(self):
        if self.prob > self.final_prob:
            self.prob -= self.anneal_rate


    def update_target_weights(self):
        sess = tf.get_default_session()
        self.target_weights = sess.run(self.var_list)


class QuestionNetwork:
    def __init__(self, ob_space, num_predictions=10, depth=20):
        self.num_predictions = num_predictions
        self.obs_size = np.prod(ob_space)
        self.x = []
        self.bs = tf.placeholder(tf.int32)
        self.initial_prediction = tf.placeholder(tf.float32, [None, num_predictions])
        prediction = self.initial_prediction
        self.predictions = [self.initial_prediction]

        # In the future, replace these single layer weights with deeper/more complex networks
        self.weight = weight_variable(shape=[self.obs_size + self.num_predictions, self.num_predictions], name='question_weight')
        self.bias = bias_variable(shape=[self.num_predictions], name='question_bias')

        for i in range(depth):
            self.x.append(tf.placeholder(tf.float32, [None] + list(ob_space)))
            a = flatten(self.x[i])
            prediction = tf.nn.sigmoid(tf.matmul(tf.concat(concat_dim=1, values=[a, prediction]),
                                                 self.weight) + self.bias)
            self.predictions.append(prediction)

        self.final_prediction = prediction

        ### adding on the top which is shared from the answer network
        self.obs = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.h = tf.placeholder(tf.float32, [None, 256])

        # get variables from answer network. Feed self.obs, self.final_prediction, self.h in and get Q out
        # using the target for Q, backpropagate through question network only
















