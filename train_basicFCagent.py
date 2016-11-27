from basicFCagent import Agent
import numpy as np
import tensorflow as tf
import random
from threading import Thread
import sys


def save_data(qnet, lossarr, prob, learn_data, weightarr, targ_weight):
    weight_avgs, avg_Q, avg_rewards, max_reward, avg_steps = qnet.test_network()
    learn_data.append([total_steps, avg_Q, avg_rewards, max_reward, avg_steps,
                          np.mean(lossarr[-100]), prob])
    weightarr.append(weight_avgs)
    np.save('learning_data', learn_data)
    np.save('weight_averages', weightarr)
    np.save('weights_' + str(int(total_steps/50000)), targ_weight)

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

while total_steps < 1000000:
    obs_sequence = np.zeros(shape=[4*UNROLLED_DEPTH])
    obs = agent.env.reset()
    done = False
    obs = obs
    steps = 0

    while not done:
        obs_sequence = np.append(obs_sequence, obs)
        obs_sequence = np.delete(obs_sequence, [0,1,2,3], axis=0)
        prob, action, reward, new_obs, _, done = agent.true_step(prob, [obs_sequence], agent.env)
        agent.update_replay_memory((obs, action, reward, new_obs, done))
        obs = new_obs

        if len(agent.replay_memory) >= 1000 and total_steps % 4 == 0:

        # Build minibatch. Should contain 32 length 10 trajectories such that the trajectory lies entirely inside
        # a single episode.
            minibatch = []
            for _ in range(32):
                buf = []
                starting_point = random.choice(range(len(agent.replay_memory) - UNROLLED_DEPTH))
                while(len(buf) < UNROLLED_DEPTH):
                    for j in range(starting_point, starting_point + UNROLLED_DEPTH):
                        buf.append(agent.replay_memory[j])
                        if agent.replay_memory[j][4]:
                            buf = []
                            starting_point = random.choice(range(len(agent.replay_memory) - UNROLLED_DEPTH))
                            break

                minibatch.append(buf)

            next_states = np.reshape([[s[3] for s in m] for m in minibatch], newshape=(32, 4*UNROLLED_DEPTH))
            action_indexes = [m[UNROLLED_DEPTH-1][1] for m in minibatch]
            rewards = [m[UNROLLED_DEPTH-1][2] for m in minibatch]
            terminals = [m[UNROLLED_DEPTH-1][4] for m in minibatch]

            feed_dict = {agent.input: next_states}
            feed_dict.update(zip(agent.weights, target_weights))
            q_vals = agent.sess.run(agent.output, feed_dict=feed_dict)
            max_q = q_vals.max(axis=1)
            target_q = np.zeros(32)
            action_list = np.zeros((32, agent.OUTPUT_SIZE))

            for i in range(32):
                target_q[i] = rewards[i]
                if not terminals[i]:
                    target_q[i] = target_q[i] + 0.99*max_q[i]
                action_list[i][action_indexes[i]] = 1.0

            states = np.reshape([[s[0] for s in m] for m in minibatch], newshape=(32, 4*UNROLLED_DEPTH))

            feed_dict = {agent.input: np.array(states), agent.target: target_q,
                         agent.action_hot: action_list}
            _, loss_val = agent.sess.run(fetches=(agent.train_operation, agent.loss), feed_dict=feed_dict)
            loss_vals.append(loss_val)


        if total_steps % 1000 == 0:
            print("updating target weights")
            target_weights = agent.sess.run(agent.weights)

        if total_steps % 3000 == 0:
            testing_thread = Thread(target=save_data, args=(agent, loss_vals, prob,
                                                            learning_data, weight_average_array, target_weights))
            testing_thread.start()

        total_steps += 1
        steps += 1

    episode_number += 1

    episode_step_count.append(steps)
    mean_steps = np.mean(episode_step_count[-100:])
    print("Training episode = {}, Total steps = {}, Last 100 mean steps = {}"
          .format(episode_number, total_steps, mean_steps))
