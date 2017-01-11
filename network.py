import tensorflow as tf
from layer_helpers import weight_variable, bias_variable



class Basic1Network:

    def __init__(self, width, activation, session=tf.get_default_session()):

        self.width = width
        self.activation = activation
        self.sess = session

        self.input = tf.placeholder(tf.float32, shape=[None, width])

        self.layer1weights = weight_variable(shape=[width, width], name="l1w")
        self.layer1bias = bias_variable(shape=[width], name="l1b")

        self.output = tf.nn.relu(tf.matmul(self.input, self.layer1weights) + self.layer1bias)

