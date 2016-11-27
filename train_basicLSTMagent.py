from basicLSTMagent import Agent
import numpy as np
import tensorflow as tf
import random
from threading import Thread
import sys

UNROLLED_DEPTH = 10

def save_data(qnet, lossarr, prob, learn_data, weightarr, targ_weight):
    weight_avgs, avg_Q, avg_rewards, max_reward, avg_steps = qnet.test_network()
    learn_data.append([total_steps, avg_Q, avg_rewards, max_reward, avg_steps,
                          np.mean(lossarr[-100]), prob])
    weightarr.append(weight_avgs)
    np.save('learning_data', learn_data)
    np.save('weight_averages', weightarr)
    np.save('weights_' + str(int(total_steps/50000)), targ_weight)

agent = Agent()


target_weights = agent.sess.run(agent.weights)
episode_step_count = []
total_steps = 1.
prob = 1.0
learning_data = []
weight_average_array = []
loss_vals = []
episode_number = 0

while total_steps < 1000000:
    obs_sequence = np.zeros(shape=[10, 4])
    obs = agent.env.reset()
    done = False
    obs = np.expand_dims(obs, axis=0)
    steps = 0

    while not done:
        obs_sequence = np.append(obs_sequence, obs, axis=0)
        obs_sequence = np.delete(obs_sequence, 0, axis=0)
        prob, action, reward, new_obs, _, done = agent.true_step(prob, obs_sequence, agent.env)
        # agent.env.render()
        agent.update_replay_memory((obs, action, reward, new_obs, done))
        obs = np.expand_dims(new_obs, axis=0)

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
                        if agent.replay_memory[j][4]:
                            buf = []
                            starting_point = random.choice(range(len(agent.replay_memory) - UNROLLED_DEPTH))
                            break

                minibatch.append(buf)

            next_states = np.reshape([[s[3] for s in m] for m in minibatch], newshape=(320, 4))
            action_indexes = [m[9][1] for m in minibatch]
            rewards = [m[9][2] for m in minibatch]
            terminals = [m[9][4] for m in minibatch]

            feed_dict = {agent.input: next_states, agent.batch_size: 32}
            q_vals = agent.sess.run(agent.output, feed_dict=feed_dict)
            max_q = q_vals.max(axis=1)
            target_q = np.zeros(32)
            action_list = np.zeros((32, agent.OUTPUT_SIZE))

            for i in range(32):
                target_q[i] = rewards[i]
                if not terminals[i]:
                    target_q[i] = target_q[i] + 0.99*max_q[i]
                action_list[i][action_indexes[i]] = 1.0

            states = np.reshape([[s[0] for s in m] for m in minibatch], newshape=(320, 4))

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
            # np.save('weights_' + str(int(total_steps/50000)), target_weights)

        total_steps += 1
        steps += 1


    episode_number += 1


    episode_step_count.append(steps)
    mean_steps = np.mean(episode_step_count[-100:])
    print("Training episode = {}, Total steps = {}, Last 100 mean steps = {}"
          .format(episode_number, total_steps, mean_steps))
