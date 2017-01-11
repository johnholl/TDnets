import tensorflow as tf
from layer_helpers import weight_variable, bias_variable

class ValueIterator:
    def __init__(self, obs_dim, max_time=4, load_path=None, replay_size=3000):
        self.replay_size = replay_size
        self.replay_memory = []
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.max_time = max_time
        self.sess = tf.Session()
        self.input_size = obs_dim
        self.output_size = 1
        self.num_recunits = 24

        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.max_time, self.input_size])
        self.rec_layer = tf.nn.rnn_cell.BasicRNNCell(self.num_recunits)
        self.in_state = self.rec_layer.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        self.rec_output, self.rec_state = tf.nn.dynamic_rnn(cell=self.rec_layer, inputs=self.input,
                                                            dtype=tf.float32, initial_state=self.in_state)


        #rec_output should have shape [batch size, max time, output size].
        self.rec_output = tf.reshape(self.rec_output, shape=[-1, self.num_recunits])

        self.l2weights = weight_variable(shape=[self.num_recunits, self.output_size], name='l2weights')
        self.l2bias = bias_variable(shape=[self.output_size], name='l2bias')

        self.output = tf.reshape(tf.matmul(self.rec_output, self.l2weights) + self.l2bias, shape=[-1, self.max_time, 1])

        self.final_value = tf.squeeze(tf.slice(self.output, begin=(0,self.max_time-1,0),
                                                    size=(-1, 1, 1)))

        self.trainable_output = tf.slice(self.output, begin=(0,int(self.max_time/2),0), size=(-1, -1, -1))

        self.targets = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.max_time - int(self.max_time/2), self.output_size])

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.trainable_output - self.targets)))#, reduction_indices=1))


        #self.optimizer = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=0.01)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.gradients_and_vars = self.optimizer.compute_gradients(self.loss)
        self.gradients = [gravar[0] for gravar in self.gradients_and_vars]
        self.vars = [gravar[1] for gravar in self.gradients_and_vars]
        self.clipped_gradients = tf.clip_by_global_norm(self.gradients, 1.)[0]
        self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.vars))

        self.saver = tf.train.Saver()

        if load_path is not None:
            self.saver.restore(self.sess, save_path=load_path)

        else:
            self.sess.run(tf.initialize_all_variables())

    def save_data(self, path):
        self.saver.save(self.sess, save_path=path)
        print("Model checkpoint saved.")


    def update_replay_memory(self, example):
        self.replay_memory.append(example)
        if len(self.replay_memory) > self.replay_size:
            self.replay_memory.pop(0)

