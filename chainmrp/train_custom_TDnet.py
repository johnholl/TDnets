import numpy as np
import time
import os
import random
import sys
from chainmrp.chain_mrp import ChainMRP, CustomChainMRP
from chainmrp.chain_answer_net import ChainAnswerNet
import tensorflow as tf

experiment_time = str(time.time())
dir = os.path.dirname(__file__)
save_path = os.path.join(dir, "checkpoints", "custom_SMTD_" + experiment_time + ".ckpt")


sess = tf.Session()
env_array = [0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1]
env = CustomChainMRP(arr=env_array)
answerNet = ChainAnswerNet(obs_dim=env.obs_dim, max_time=6, scope="TDnet", softmax=True, depth=6)
masklength = int(answerNet.max_time/2)


steps = 0
loss_vals = []
episode_num = 0

while steps < 300000:
    obs_sequence = np.zeros(shape=[answerNet.max_time, answerNet.input_size])
    obs = env.reset(env.get_random_state())
    obs_sequence = np.append(obs_sequence, [obs], axis=0)
    obs_sequence = np.delete(obs_sequence, [0], axis=0)
    new_obs_sequence = obs_sequence
    episode_num += 1
    done = False

    while not done:
        new_obs, reward, term = env.step()
        new_obs_sequence = np.append(new_obs_sequence, [new_obs], axis=0)
        new_obs_sequence = np.delete(new_obs_sequence, [0], axis=0)
        answerNet.update_replay_memory(example=(obs_sequence, reward, new_obs_sequence, done))
        obs = new_obs
        obs_sequence = np.append(obs_sequence, [new_obs], axis=0)
        obs_sequence = np.delete(obs_sequence, [0], axis=0)
        done = term
        steps +=1

        if len(answerNet.replay_memory) > 200:
            minibatch = random.sample(answerNet.replay_memory, k=32)
            obs_sequences = np.array([m[0] for m in minibatch])
            next_obs_sequences = np.array([m[2] for m in minibatch])

            feed_dict = {answerNet.input: next_obs_sequences, answerNet.batch_size: 32}
            lookahead_predictions = answerNet.sess.run(answerNet.normalized_rec_output, feed_dict=feed_dict)
            question_input = np.concatenate((next_obs_sequences, lookahead_predictions), axis=2)
            prediction_targets = np.zeros(shape=[32, masklength, answerNet.questionnet.depth*answerNet.questionnet.obs_dim])
            for i in range(masklength):
                prediction_targets[:, i, :] = answerNet.sess.run(answerNet.questionnet.output,
                                             feed_dict={answerNet.questionnet.input: question_input[:, i+masklength, :]})

            feed_dict = {answerNet.input: obs_sequences, answerNet.pred_targets: prediction_targets,
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