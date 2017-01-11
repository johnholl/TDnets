import numpy as np
import time
import os
import random
import sys
from chainmrp.chain_mrp import ChainMRP, CustomChainMRP
from chainmrp.chain_answer_net import ChainAnswerNet, DiscountChainAnswerNet, BasicDiscountChainAnswerNet, MultilayerRecurrentAgent
import tensorflow as tf

experiment_time = str(time.time())
dir = os.path.dirname(__file__)
save_path = os.path.join(dir, "checkpoints", "fulldiscount_multilayerRecurrent_model" + experiment_time + ".ckpt")


env = ChainMRP(prob=1., n=10, k=4)
# env = CustomChainMRP(arr=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
answerNet = MultilayerRecurrentAgent(max_time=6, obs_dim=env.obs_dim, discount=env.discount)
masklength = answerNet.masklength

minibatch_size = 32
steps = 0
loss_vals = []
episode_num = 0
target_weights = answerNet.sess.run(answerNet.weights)
target_update_frequency = 4000

while steps < 200000:
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
        answerNet.update_replay_memory(example=(obs_sequence, reward, new_obs_sequence, term))
        obs = new_obs
        obs_sequence = np.append(obs_sequence, [new_obs], axis=0)
        obs_sequence = np.delete(obs_sequence, [0], axis=0)
        done = term
        steps +=1

        if len(answerNet.replay_memory) > 200:
            minibatch = random.sample(answerNet.replay_memory, k=minibatch_size)
            obs_sequences = np.array([m[0] for m in minibatch])
            next_obs_sequences = np.array([m[2] for m in minibatch])
            terminals = np.array([m[3] for m in minibatch])

            feed_dict = {answerNet.input: next_obs_sequences, answerNet.batch_size: minibatch_size}
            feed_dict.update(zip(answerNet.weights, target_weights))
            lookahead_predictions = answerNet.sess.run(answerNet.rec_output, feed_dict=feed_dict)
            question_input = np.concatenate((next_obs_sequences[:,:,:], lookahead_predictions), axis=2)
            prediction_targets = np.zeros(shape=[minibatch_size, masklength, answerNet.questionnet.obs_dim])
            for i in range(masklength):
                prediction_targets[:, i, :] = answerNet.sess.run(answerNet.questionnet.output,
                                             feed_dict={answerNet.questionnet.input: question_input[:, i+masklength, :]})
            for j in range(minibatch_size):
                if terminals[j]:
                    prediction_targets[j, -1,:] = np.concatenate((np.zeros(shape=(env.obs_dim -1)) , [answerNet.discount]))

            #
            # print(obs_sequences)
            # print(next_obs_sequences)
            # print(lookahead_predictions)
            # print(question_input)
            # print(prediction_targets)
            # a = input("Enter to continue")
            feed_dict = {answerNet.input: obs_sequences, answerNet.pred_targets: prediction_targets,
                          answerNet.batch_size: minibatch_size}
            _, loss_val = answerNet.sess.run(fetches=(answerNet.train_operation, answerNet.loss), feed_dict=feed_dict)
            loss_vals.append(loss_val)

        if steps % target_update_frequency == 0:
            target_weights = answerNet.sess.run(answerNet.weights)
            print("updated target weights")

        if steps % 1000 == 0:
            answerNet.save_data(path=save_path)
            pass


    avg_loss = np.mean(loss_vals[-100:])
    print("Most recent average loss: ", avg_loss)
    print(steps)
    # np.save(os.path.join(dir, "loss_" + experiment_time), loss_vals)