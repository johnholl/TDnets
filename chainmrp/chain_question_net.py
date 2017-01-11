import tensorflow as tf
import numpy as np


class ChainQuestionNet:

    def __init__(self, obs_dim, depth, graph):
        with graph.as_default():
            self.depth = depth
            self.input_size = obs_dim * (depth+1)
            self.output_size = obs_dim * depth
            self.obs_dim = obs_dim

            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size])

            self.weight_matrix = np.eye(N=self.input_size, M=self.output_size, dtype=np.float32)
            self.weights = tf.constant(self.weight_matrix)

            self.output = tf.matmul(self.input, self.weights)

# from chainmrp.chain_mrp import ChainMRP
# env = ChainMRP()
# print(env.obs_dim)
# cqn = ChainQuestionNet(obs_dim=env.obs_dim, depth=4)


class DiscountQuestionNet:

    def __init__(self, obs_dim, graph, gamma=.99):
        with graph.as_default():
            self.input_size = obs_dim*2
            self.output_size = obs_dim
            self.obs_dim = obs_dim

            self.input = tf.placeholder(dtype=tf.float64, shape=[None, self.input_size])

            self.weight_mat = np.concatenate((np.eye(self.obs_dim, self.obs_dim), gamma*np.eye(self.obs_dim, self.obs_dim)))
            self.weights = tf.constant(self.weight_mat)

            self.output = tf.matmul(self.input, self.weights)


# from chainmrp.chain_mrp import ChainMRP
# env = ChainMRP()
# sess = tf.Session()
# dqn = DiscountQuestionNet(obs_dim=env.obs_dim, sess=sess)
# print(dqn.weight_matrix)
