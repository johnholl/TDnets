import numpy as np
import time
import os
import random
import sys
from chainmrp.chain_mrp import ChainMRP
from chainmrp.chain_answer_net import ChainAnswerNet
import tensorflow as tf

experiment_time = str(time.time())
dir = os.path.dirname(__file__)
save_path = os.path.join(dir, "checkpoints", "supervisedmodel_" + experiment_time + ".ckpt")

env = ChainMRP(prob=1., n=10)
sess = tf.Session()
answerNet = ChainAnswerNet(obs_dim=env.obs_dim, max_time=10)
masklength = int(answerNet.max_time/2)


steps = 0
loss_vals = []
episode_num = 0

while steps < 200000:
    obs_sequence = np.zeros(shape=[answerNet.max_time + answerNet.questionnet.depth, answerNet.input_size])
    obs = env.reset(env.get_random_state())
    reward = 0

    episode_num += 1
    done = False

    while not done:
        obs_sequence = np.append(obs_sequence, [obs], axis=0)
        obs_sequence = np.delete(obs_sequence, [0], axis=0)
        answerNet.update_replay_memory(example=(obs_sequence, reward, done))
        new_obs, reward, term = env.step()
        obs = new_obs
        steps +=1
        done = term

        if len(answerNet.replay_memory) > 200:
            minibatch = random.sample(answerNet.replay_memory, k=32)
            obs_sequences = np.array([m[0] for m in minibatch])
            batch_target = []
            for i in range(32):
                entry = np.array(minibatch[i][0])   # shape= (14, 11)
                single_target = []
                for j in range(answerNet.masklength, answerNet.max_time):
                    t = np.ndarray.flatten(entry[j+1:j+answerNet.questionnet.depth+1])
                    single_target.append(t)
                batch_target.append(single_target)

            states = obs_sequences[:, :answerNet.max_time, :]

            feed_dict = {answerNet.input: np.array(states), answerNet.pred_targets: batch_target,
                          answerNet.batch_size: 32}
            _, loss_val = answerNet.sess.run(fetches=(answerNet.train_operation, answerNet.loss), feed_dict=feed_dict)
            loss_vals.append(loss_val)

        if steps % 5000 == 0:
            # answerNet.save_data(path=save_path)
            pass


    avg_loss = np.mean(loss_vals[-100:])
    print("Most recent average loss: ", avg_loss)
    print(steps)
    # np.save(os.path.join(dir, "loss_" + experiment_time), loss_vals)
