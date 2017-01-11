from chainmrp.chain_mrp import ChainMRP, CustomChainMRP
from chainmrp.chain_answer_net import ChainAnswerNet, DiscountChainAnswerNet, BasicDiscountChainAnswerNet, RecurrentAgent, MultilayerRecurrentAgent
import os
import numpy as np
import tensorflow as tf
import sys

# dir = os.path.dirname(__file__)
# path = os.path.join(dir, "checkpoints", "model_1482423718.9684966.ckpt")
loadpath = \
"/home/john/code/pythonfiles/TDnets/chainmrp/checkpoints/fulldiscount_multilayerRecurrent_model1483978030.5397859.ckpt"
env = ChainMRP(n=10, k=4, prob=1.)
# env = CustomChainMRP(arr=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
# answerNet = ChainAnswerNet(
#             load_path="/home/john/code/pythonfiles/TDnets/chainmrp/checkpoints/TDmodel_targ_1483641835.487793.ckpt",
#                            obs_dim=env.obs_dim, softmax=False)

# answerNet = ChainAnswerNet(scope="Supervised",
#             load_path="/home/john/code/pythonfiles/TDnets/chainmrp/checkpoints/supervisedmodel_1483551517.6653957.ckpt",
#             obs_dim=env.obs_dim, softmax=False)

# answerNet = BasicDiscountChainAnswerNet(
#             load_path="/home/john/code/pythonfiles/TDnets/chainmrp/checkpoints/discount_TDmodel_1483646829.004575.ckpt",
#             obs_dim=env.obs_dim, max_time=6)

# answerNet = DiscountChainAnswerNet(
#             load_path=loadpath,
#             obs_dim=env.obs_dim, max_time=6)

# answerNet = RecurrentAgent(load_path=loadpath, obs_dim=env.obs_dim, max_time=6, discount=env.discount)

answerNet = MultilayerRecurrentAgent(load_path=loadpath, obs_dim=env.obs_dim, max_time=6, discount=env.discount)

steps = 0
episode_num = 0
depth_loss_arr = []
rounded_depth_loss_arr = []
loss_arr = []
rounded_loss_arr = []
full_knowledge_loss_arr = []
full_knowledge_rounded_loss_arr = []

while steps < 10000:
    obs_sequence = np.zeros(shape=[answerNet.max_time, answerNet.input_size])
    obs = env.reset(env.get_random_state())
    obs_sequence = np.append(obs_sequence, [obs], axis=0)
    obs_sequence = np.delete(obs_sequence, [0], axis=0)
    episode_num += 1
    done = False
    term = False
    state = env.state


    while not done:
        print(obs_sequence)
        predictions = answerNet.sess.run(answerNet.final_prediction, feed_dict={answerNet.input: [obs_sequence], answerNet.batch_size: 1})
        # predictions = np.reshape(predictions, newshape=[1, answerNet.questionnet.obs_dim])
        rounded_predictions = np.round(predictions)
        absolute_predictions = np.argmax(predictions, axis=1)
        print("Predictions = ", predictions)
        # print("Absolute Predictions = ", absolute_predictions)
        # print("State = ", state)


        # if state + answerNet.depth < env.num_states - 1:
        #     actual = env.get_next_k_obs(answerNet.depth)
        #     depth_loss = np.sum(np.square(actual-predictions), axis=1)
        #     rounded_depth_loss = np.sum(np.square(actual-rounded_predictions), axis=1)
        #     loss = np.sum(depth_loss)
        #     rounded_loss = np.sum(rounded_depth_loss)
        #     depth_loss_arr.append(depth_loss)
        #     rounded_depth_loss_arr.append(rounded_depth_loss)
        #     loss_arr.append(loss)
        #     rounded_loss_arr.append(rounded_loss)
        #     if 1 in obs_sequence[:, 1:]:
        #         full_knowledge_loss_arr.append(depth_loss)
        #         full_knowledge_rounded_loss_arr.append(rounded_depth_loss)


        # actual = env.get_all_discounted_sums(gamma=.99)
        actual = env.get_discounted_sum(gamma=.99)
        loss = np.square(predictions - actual)
        loss_arr.append(loss)

        a = input("Press Enter to continue")
        new_obs, reward, done = env.step()
        obs = new_obs
        obs_sequence = np.append(obs_sequence, [obs], axis=0)
        obs_sequence = np.delete(obs_sequence, [0], axis=0)
        steps +=1
        state = env.state
#
avg_loss = np.average(loss_arr)
# avg_depth_loss = np.average(depth_loss_arr, axis=0)
# avg_rounded_loss = np.average(rounded_loss_arr)
# avg_rounded_depth_loss = np.average(rounded_depth_loss_arr, axis=0)
# avg_full_knowledge_loss = np.average(full_knowledge_loss_arr, axis=0)
# avg_full_knowledge_rounded_loss = np.average(full_knowledge_rounded_loss_arr, axis=0)
#
print(avg_loss)
# print(avg_depth_loss)
# print(avg_rounded_loss)
# print(avg_rounded_depth_loss)
# print(avg_full_knowledge_loss)
# print(avg_full_knowledge_rounded_loss)
