from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops.rnn_cell import RNNCell, _linear, tanh
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf



class SoftMaxRNNCell(RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated." % self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      output = tf.nn.softmax(self._activation(_linear([inputs, state], self._num_units, True)))
    return output, output


