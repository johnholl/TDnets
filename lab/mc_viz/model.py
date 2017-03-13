import tensorflow as tf
from layer_helpers import conv2d, flatten, normalized_columns_initializer, linear, categorical_sample
from custom_rnn_cells import GridPredictionLSTMCell, PredictionLSTMStateTuple, BasicPixelRNN
from rnn import old_dynamic_rnn
import numpy as np


# should include policies for
# (1) behavior agent and deterministic behavior agent
# (2) monte carlo baseline network
# (3) bptt agent. Similar to Q_PC but will only have 1 prediction per value (state value instead of action value)
# (4) nonbptt agent. Similar to (3) but uses prediction placeholder instead of prediction recurrent cell


###### (1) BEHAVIOR AGENT ###########################


class BehaviorAgent(object):
    def __init__(self, ob_space, ac_space, sess, replay_size=2000, grid_size=20, ckpt_file=None):
        with tf.variable_scope("global"):
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
            self.action = tf.placeholder(tf.float32, [None, ac_space])
            self.reward = tf.placeholder(tf.float32, [None, 1])
            self.bs = tf.placeholder(dtype=tf.int32)
            self.replay_memory = []
            self.replay_size = replay_size
            self.grid_size = grid_size

            x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
            x = conv_features = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
            x = flatten(x)
            x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
            x = tf.concat(axis=1, values=[x, self.action, self.reward])
            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(x, [0])

            size = 256
            lstm = GridPredictionLSTMCell(size, state_is_tuple=True, ac_space=ac_space,
                                           grid_size=20)
            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            pred_init = np.zeros((1, lstm.state_size.pred), np.float32)
            self.state_init = [c_init, h_init, pred_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            pred_in = tf.placeholder(tf.float32, [1, lstm.state_size.pred])
            self.state_in = [c_in, h_in, pred_in]

            state_in = PredictionLSTMStateTuple(c_in, h_in, pred_in)
            lstm_outputs, lstm_state = old_dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h, lstm_pred = lstm_state
            x = tf.reshape(lstm_outputs, [-1, size])

            # Actor critic branch
            self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :], lstm_pred[:1, :]]
            self.sample = categorical_sample(self.logits, ac_space)[0, :]

            # Auxiliary branch
            self.predictions = tf.reshape(lstm_pred, shape=[-1, grid_size, grid_size, ac_space])
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            self.target_weights = []

            self.ckpt_file = ckpt_file


    def get_initial_features(self):
        return self.state_init

    def act(self, ob, prev_a, prev_r, c, h, pred, current_session=None):
        sess = tf.get_default_session()
        if current_session is not None:
            sess = current_session

        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                         self.state_in[0]: c, self.state_in[1]: h,
                         self.state_in[2]: pred})

    def value(self, ob, prev_a, prev_r, c, h, pred):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                  self.state_in[0]: c, self.state_in[1]: h,
                                  self.state_in[2]: pred})[0]

    def update_replay_memory(self, tuple):
        # appends tuple and pops old tuple if memory size is exceeded
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)

    def update_target_weights(self):
        sess = tf.get_default_session()
        self.target_weights = sess.run(self.var_list)

    def restore(self, sess, saver):
        if self.ckpt_file is not None:
            saver.restore(sess, self.ckpt_file)


class DeterministicBehaviorAgent(object):
    def __init__(self, ob_space, ac_space, sess, replay_size=2000, grid_size=20, ckpt_file=None):
        with tf.variable_scope("global"):
            self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
            self.action = tf.placeholder(tf.float32, [None, ac_space])
            self.reward = tf.placeholder(tf.float32, [None, 1])
            self.bs = tf.placeholder(dtype=tf.int32)
            self.replay_memory = []
            self.replay_size = replay_size
            self.grid_size = grid_size

            x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
            x = conv_features = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
            x = flatten(x)
            x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
            x = tf.concat(axis=1, values=[x, self.action, self.reward])
            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(x, [0])

            size = 256
            lstm = GridPredictionLSTMCell(size, state_is_tuple=True, ac_space=ac_space,
                                           grid_size=20)
            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            pred_init = np.zeros((1, lstm.state_size.pred), np.float32)
            self.state_init = [c_init, h_init, pred_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            pred_in = tf.placeholder(tf.float32, [1, lstm.state_size.pred])
            self.state_in = [c_in, h_in, pred_in]

            state_in = PredictionLSTMStateTuple(c_in, h_in, pred_in)
            lstm_outputs, lstm_state = old_dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h, lstm_pred = lstm_state
            x = tf.reshape(lstm_outputs, [-1, size])

            # Actor critic branch
            self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
            self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :], lstm_pred[:1, :]]
            self.sample = tf.one_hot(tf.squeeze(tf.argmax(self.logits, axis=1)), ac_space)

            # Auxiliary branch
            self.predictions = tf.reshape(lstm_pred, shape=[-1, grid_size, grid_size, ac_space])
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            self.target_weights = []

            self.ckpt_file = ckpt_file


    def get_initial_features(self):
        return self.state_init

    def act(self, ob, prev_a, prev_r, c, h, pred, current_session=None):
        sess = tf.get_default_session()
        if current_session is not None:
            sess = current_session

        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                         self.state_in[0]: c, self.state_in[1]: h,
                         self.state_in[2]: pred})

    def value(self, ob, prev_a, prev_r, c, h, pred):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                  self.state_in[0]: c, self.state_in[1]: h,
                                  self.state_in[2]: pred})[0]

    def update_replay_memory(self, tuple):
        # appends tuple and pops old tuple if memory size is exceeded
        self.replay_memory.append(tuple)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)

    def update_target_weights(self):
        sess = tf.get_default_session()
        self.target_weights = sess.run(self.var_list)

    def restore(self, sess, saver):
        if self.ckpt_file is not None:
            saver.restore(sess, self.ckpt_file)



######## (2) MONTE CARLO BASELINE NETWORK ##############
class MonteCarloValue(object):
    def __init__(self, ob_space, ac_space, grid_size=20):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.pred = tf.placeholder(tf.float32, [None, grid_size, grid_size, 1])

        self.bs = tf.placeholder(dtype=tf.int32)
        self.replay_memory = []
        self.grid_size = grid_size

        self.num_actions = ac_space

        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
        x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
        x = flatten(x)
        pred = flatten(self.pred)
        x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
        x = tf.concat(axis=1, values=[x, self.action, self.reward, pred])

        x = tf.nn.relu(linear(x, 256, "l4", normalized_columns_initializer(0.1)))

        val = linear(x, 1, "value", normalized_columns_initializer(0.01))
        self.val = tf.reshape(val, shape=[-1])

        # Auxiliary branch
        y = linear(x, 32*(self.grid_size-10)*(self.grid_size-10), 'auxbranch', normalized_columns_initializer(0.1))
        y = tf.reshape(y, shape=[-1, self.grid_size-10, self.grid_size-10, 32])
        deconv_weights = tf.get_variable("deconv" + "/w", [4, 4, 1, 32])
        predictions = tf.nn.conv2d_transpose(y, deconv_weights,
                                                output_shape=[1, self.grid_size, self.grid_size, 1],
                                                strides=[1,2,2,1], padding='SAME')

        self.predictions = tf.reshape(predictions, shape=[-1, grid_size*grid_size])

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)


    def value(self, ob, prev_a, prev_r, pred):
        sess = tf.get_default_session()
        reward_value, prediction_values = sess.run((self.val, self.predictions), {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                  self.pred: pred})

        return reward_value, prediction_values


######## (3) BPTT AGENT ################################
class BPTTV_PCR(object):
    def __init__(self, ob_space, ac_space, grid_size=20):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.grid_size = grid_size

        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
        x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
        x = flatten(x)
        x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
        x = tf.concat(axis=1, values=[x, self.action, self.reward])
        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(x, [0])

        size = 256
        # replace with a basic grid prediction rnn cell
        lstm = BasicPixelRNN(size, grid_size=20)
        self.state_size = size
        step_size = tf.shape(self.x)[:1]

        pred_init = np.zeros((1, 20*20), np.float32)
        self.state_init = pred_init
        pred_in = tf.placeholder(tf.float32, [1, 20*20])
        self.state_in = pred_in

        outputs, predictions = tf.nn.dynamic_rnn(
            lstm, x, initial_state=self.state_in, sequence_length=step_size,
            time_major=False)

        # Q learning branch
        x = tf.reshape(outputs, [-1, size])
        val = linear(x, 1, "action", normalized_columns_initializer(0.01))
        self.val = tf.reshape(val, shape=[-1])
        # self.state_out = [lstm_c[:1, :], lstm_h[:1, :], lstm_pred[:1, :]]

        # Auxiliary branch
        self.predictions = tf.reshape(predictions, shape=[-1, grid_size, grid_size])
        self.unshaped_predictions = predictions
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def value(self, ob, prev_a, prev_r, pred):
        sess = tf.get_default_session()
        reward_value, prediction_values = sess.run((self.val, self.unshaped_predictions), {self.x: [ob], self.action: [prev_a], self.reward: [[prev_r]],
                                  self.state_in: pred})

        return reward_value, prediction_values

######## (4) NON-BPTT AGENT ############################

class FFVPCR_Policy(object):
    def __init__(self, ob_space, ac_space, grid_size=20):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        self.action = tf.placeholder(tf.float32, [None, ac_space])
        self.reward = tf.placeholder(tf.float32, [None, 1])
        self.pred = tf.placeholder(tf.float32, [None, grid_size*grid_size])
        self.bs = tf.placeholder(dtype=tf.int32)

        self.grid_size = grid_size

        self.num_actions = ac_space

        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))
        x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))
        x = flatten(x)
        x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
        x = tf.concat(axis=1, values=[x, self.action, self.reward, self.pred])

        x = tf.nn.relu(linear(x, 256, "l4", normalized_columns_initializer(0.1)))

        val = linear(x, 1, "value", normalized_columns_initializer(0.01))
        self.val = tf.reshape(val, shape=[-1])

        # Auxiliary branch
        y = linear(x, 32*(self.grid_size-10)*(self.grid_size-10), 'auxbranch', normalized_columns_initializer(0.1))
        self.testy = y = tf.reshape(y, shape=[-1, self.grid_size-10, self.grid_size-10, 32])
        deconv_weights = tf.get_variable("deconv" + "/w", [4, 4, 1, 32])
        predictions = tf.nn.conv2d_transpose(y, deconv_weights,
                                                output_shape=[self.bs, self.grid_size, self.grid_size, 1],
                                                strides=[1,2,2,1], padding='SAME')

        self.predictions = tf.reshape(predictions, shape=[-1, grid_size*grid_size])

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return np.zeros((1, self.grid_size*self.grid_size), np.float32)

    def value(self, ob, prev_a, prev_r, pred):
        sess = tf.get_default_session()
        reward_value, prediction_values = sess.run((self.val, self.predictions), {self.x: [ob], self.action: [prev_a],
                                                                                  self.reward: [[prev_r]], self.bs:1,
                                                                                  self.pred: pred})

        return reward_value, prediction_values
