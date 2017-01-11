import tensorflow as tf
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops.rnn_cell import _linear, RNNCell
from tensorflow.python.ops import variable_scope as vs

from network import Basic1Network


class CustomRnnCell(RNNCell):

    def __init__(self, num_units, input_size=None, activation=tanh):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated." % self)
        self._num_units = num_units
        self._activation = activation
        self.network = Basic1Network(num_units, activation=activation)

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            self.network.inputs = inputs
            output = self.network.output
            return output, output

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units


class MultilayerRNNCell(RNNCell):
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
            layer1 = self._activation(_linear([inputs, state], self._num_units, True, scope="firstlayer"))
            output = self._activation(_linear([layer1], self._num_units, True, scope="secondlayer"))
            # state = tf.concat(concat_dim=1, values=(layer1, output))
        return output, output