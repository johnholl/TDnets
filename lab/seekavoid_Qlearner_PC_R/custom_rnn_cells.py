from tensorflow.python.platform import tf_logging as logging
# from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell import RNNCell, _linear
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
import collections
import tensorflow as tf
import numpy as np


class GridPredictionLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, ac_space, grid_size=20, forget_bias=1.0, input_size=None,
               state_is_tuple=True, activation=tanh):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 3-tuples of
        the `c_state`, `h_state`, and `pred_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._num_predictions = grid_size*grid_size*ac_space
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self.ac_space = ac_space
    self.grid_size = grid_size

  @property
  def state_size(self):
    return (PredictionLSTMStateTuple(self._num_units, self._num_units, self._num_predictions)
            if self._state_is_tuple else 2 * self._num_units + self._num_predictions)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h, pred = state
      else:
        c, h, pred = array_ops.split(1, 3, state)

      z = tf.nn.l2_normalize(pred, dim=1)
      z = tf.nn.tanh(linear(z, 256, 'encode_pred', normalized_columns_initializer(0.1)))
      # z = tf.constant(value=0., dtype=tf.float32, shape=[1, 256], name='encode_pred')
      concat = _linear([inputs, h, z], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(1, 4, concat)

      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)

      ## Now, from the new_c compute a new prediction

      y = linear(new_c, 32*(self.grid_size-10)*(self.grid_size-10), 'auxbranch', normalized_columns_initializer(0.1))
      y = tf.reshape(y, shape=[-1, self.grid_size-10, self.grid_size-10, 32])
      deconv_weights = tf.get_variable("deconv" + "/w", [4, 4, self.ac_space, 32])
      new_pred_unshaped = tf.nn.conv2d_transpose(y, deconv_weights,
                                                output_shape=[1, self.grid_size, self.grid_size, self.ac_space],
                                                strides=[1,2,2,1], padding='SAME')
      new_pred = tf.reshape(new_pred_unshaped, shape=[1, self._num_predictions])


      if self._state_is_tuple:
        new_state = PredictionLSTMStateTuple(new_c, new_h, new_pred)
      else:
        new_state = array_ops.concat(1, [new_c, new_h, new_pred])
      return new_h, new_state


_PredictionLSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h", "pred"))

class PredictionLSTMStateTuple(_PredictionLSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores three elements: `(c, h, pred)`, in that order.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h, pred) = self
    if not c.dtype == h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s vs %s" %
                      (str(c.dtype), str(h.dtype), str(pred.dtype)))
    return c.dtype



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


def weight_variable(shape, initial_weight=None):
    if initial_weight is None:
        initial = tf.random_normal(shape, stddev=0.01)
        return tf.get_variable(initial)
    else:
        return tf.get_variable(initial_weight)



####################################################################

class GridConvPredictionLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, ac_space, grid_size=20, forget_bias=1.0, input_size=None,
               state_is_tuple=True, activation=tanh):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 3-tuples of
        the `c_state`, `h_state`, and `pred_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._num_predictions = grid_size*grid_size*ac_space
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self.ac_space = ac_space
    self.grid_size = grid_size

  @property
  def state_size(self):
    return (PredictionLSTMStateTuple(self._num_units, self._num_units, self._num_predictions)
            if self._state_is_tuple else 2 * self._num_units + self._num_predictions)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h, pred = state
      else:
        c, h, pred = array_ops.split(1, 3, state)

      # Prediction should be [1,3200]. Reshape to [1,20,20,8], perform convolution, then flatten

      z = tf.nn.l2_normalize(pred, dim=1)
      z = tf.nn.tanh(linear(z, 256, 'encode_pred', normalized_columns_initializer(0.1)))
      # z = tf.constant(value=0., dtype=tf.float32, shape=[1, 256], name='encode_pred')
      concat = _linear([inputs, h, z], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(1, 4, concat)

      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)

      ## Now, from the new_c compute a new prediction

      y = linear(new_c, 32*(self.grid_size-10)*(self.grid_size-10), 'auxbranch', normalized_columns_initializer(0.1))
      y = tf.reshape(y, shape=[-1, self.grid_size-10, self.grid_size-10, 32])
      deconv_weights = tf.get_variable("deconv" + "/w", [4, 4, self.ac_space, 32])
      new_pred_unshaped = tf.nn.conv2d_transpose(y, deconv_weights,
                                                output_shape=[1, self.grid_size, self.grid_size, self.ac_space],
                                                strides=[1,2,2,1], padding='SAME')
      new_pred = tf.reshape(new_pred_unshaped, shape=[1, self._num_predictions])


      if self._state_is_tuple:
        new_state = PredictionLSTMStateTuple(new_c, new_h, new_pred)
      else:
        new_state = array_ops.concat(1, [new_c, new_h, new_pred])
      return new_h, new_state


_PredictionLSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h", "pred"))

