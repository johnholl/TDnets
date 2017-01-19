from basicLSTMagent import Agent
import numpy as np
import tensorflow as tf
import random
from threading import Thread
import time
import sys

time = time.time()

def save_data(qnet, lossarr, prob, learn_data, weightarr, targ_weight):
    weight_avgs, avg_Q, avg_rewards, max_reward, avg_steps = qnet.test_network()
    learn_data.append([total_steps, avg_Q, avg_rewards, max_reward, avg_steps,
                       np.mean(lossarr[-100]), prob])
    weightarr.append(weight_avgs)
    np.save('learning_data_lstm' + str(time), learn_data)
    np.save('weight_averages' + str(time), weightarr)
    # np.save('weights_' + str(int(total_steps/3000)) + "_" + str(time), targ_weight)

agent = Agent()
UNROLLED_DEPTH = agent.MAX_DEPTH


target_weights = agent.sess.run(agent.weights)
episode_step_count = []
total_steps = 1.
prob = 1.0
learning_data = []
weight_average_array = []
loss_vals = []
episode_number = 0

while total_steps < 4000000:
    obs_sequence = np.zeros(shape=[UNROLLED_DEPTH, 4, 5])
    obs = agent.env.reset()
    obs_sequence = np.append(obs_sequence, [obs], axis=0)
    obs_sequence = np.delete(obs_sequence, [0], axis=0)
    done = False
    steps = 0

    while not done:
        prob, action, reward, new_obs, _, done = agent.true_step(prob, obs_sequence.reshape([UNROLLED_DEPTH, 20]), agent.env)
        agent.update_replay_memory((obs, action, reward, new_obs, done))
        obs = new_obs
        obs_sequence = np.append(obs_sequence, [obs], axis=0)
        obs_sequence = np.delete(obs_sequence, [0], axis=0)

        if len(agent.replay_memory) >= 1000 and total_steps % 4 == 0:

        # Build minibatch. Should contain 32 length 10 trajectories such that the trajectory lies entirely inside
        # a single episode.
            minibatch = []
            for _ in range(32):
                buf = []
                starting_point = random.choice(range(len(agent.replay_memory) - UNROLLED_DEPTH))
                while(len(buf) < 10):
                    for j in range(starting_point, starting_point + UNROLLED_DEPTH):
                        buf.append(agent.replay_memory[j])
                        if(agent.replay_memory[j][4] and len(buf)<10):
                            buf = []
                            starting_point = random.choice(range(len(agent.replay_memory) - UNROLLED_DEPTH))
                            break

                minibatch.append(buf)

            next_states = np.reshape([[s[3] for s in m] for m in minibatch], newshape=(320, 20))

            action_indexes = np.reshape([[s[1] for s in m] for m in minibatch], newshape=(320))
            rewards = np.reshape([[s[2] for s in m] for m in minibatch], newshape=(320))
            terminals = np.reshape([[s[4] for s in m] for m in minibatch], newshape=(320))

            feed_dict = {agent.input: next_states, agent.batch_size: 32}
            q_vals = agent.sess.run(agent.output, feed_dict=feed_dict)
            max_q = q_vals.max(axis=1)
            target_q = np.zeros(320)
            action_list = np.zeros((320, agent.OUTPUT_SIZE))

            for i in range(320):
                target_q[i] = rewards[i]
                if not terminals[i]:
                    target_q[i] = target_q[i] + 0.99*max_q[i]
                action_list[i][action_indexes[i]] = 1.0

            states = np.reshape([[s[0] for s in m] for m in minibatch], newshape=(320, 20))

            feed_dict = {agent.input: np.array(states), agent.target: target_q,
                         agent.action_hot: action_list, agent.batch_size: 32}
            _, loss_val = agent.sess.run(fetches=(agent.train_operation, agent.loss), feed_dict=feed_dict)
            loss_vals.append(loss_val)


        if total_steps % 1000 == 0:
            print("updating target weights")
            target_weights = agent.sess.run(agent.weights)

        if total_steps % 3000 == 0:
            testing_thread = Thread(target=save_data, args=(agent, loss_vals, prob,
                                                            learning_data, weight_average_array, target_weights))
            testing_thread.start()

            # weight_avgs, avg_Q, avg_rewards, max_reward, avg_steps = dqn.test_network()
            # learning_data.append([total_steps, avg_Q, avg_rewards, max_reward, avg_steps,
            #                       np.mean(loss_vals[-100]), prob])
            # weight_average_array.append(weight_avgs)
            # np.save('learning_data', learning_data)
            # np.save('weight_averages', weight_average_array)
            # np.save(
            # 'weights_' + str(int(total_steps/50000)), target_weights)

        total_steps += 1
        steps += 1


    episode_number += 1


    episode_step_count.append(steps)
    mean_steps = np.mean(episode_step_count[-100:])
    print("Training episode = {}, Total steps = {}, Last 100 mean steps = {}"
          .format(episode_number, total_steps, mean_steps))
