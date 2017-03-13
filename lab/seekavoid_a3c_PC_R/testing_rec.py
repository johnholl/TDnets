from custom_rnn_cells import testBasicLSTMCell, PredictionLSTMStateTuple, GridPredictionLSTMCell
from model import conv2d, flatten, normalized_columns_initializer
import tensorflow as tf
import numpy as np


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


x = tf.placeholder(tf.float32, [None, 84, 84, 3])
bs = tf.placeholder(dtype=tf.int32)
x = flatten(x)
x = tf.nn.relu(linear(x, 256, "l3", normalized_columns_initializer(0.1)))
# introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
x = tf.expand_dims(x, [0])

lstm = GridPredictionLSTMCell(num_units=20, ac_space=8, bs=bs)
state_size = lstm.state_size
step_size = tf.shape(x)[:1]

c_init = np.zeros((1, lstm.state_size.c), np.float32)
h_init = np.zeros((1, lstm.state_size.h), np.float32)
pred_init = np.zeros((1, lstm.state_size.pred), np.float32)
state_init = [c_init, h_init, pred_init]
c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
pred_in = tf.placeholder(tf.float32, [1, lstm.state_size.pred])
state_in_tuple = [c_in, h_in, pred_in]

state_in = PredictionLSTMStateTuple(c_in, h_in, pred_in)
lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
    lstm, x, initial_state=state_in, sequence_length=step_size,
    time_major=False)
lstm_c, lstm_h, lstm_pred = lstm_state
x = tf.reshape(lstm_outputs, [-1, 20])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
