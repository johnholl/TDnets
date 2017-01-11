from basicFCagent import Agent
import numpy as np
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
    np.save('learning_data' + str(time), learn_data)
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

while total_steps < 2000000:
    obs_sequence = np.zeros(shape=[UNROLLED_DEPTH, 4, 5])
    obs = agent.env.reset()
    obs_sequence = np.append(obs_sequence, [obs], axis=0)
    obs_sequence = np.delete(obs_sequence, [0], axis=0)
    done = False
    steps = 0

    while not done:
        prob, action, reward, new_obs, _, done = agent.true_step(prob, [obs_sequence.flatten()], agent.env)
        new_obs_sequence = np.append(obs_sequence, [new_obs], axis=0)
        new_obs_sequence = np.delete(new_obs_sequence, [0], axis=0)
        agent.update_replay_memory((obs_sequence.flatten(), action, reward, new_obs_sequence.flatten(), done))
        obs = new_obs
        obs_sequence = new_obs_sequence

        if len(agent.replay_memory) >= 1000 and total_steps % 4 == 0:
            minibatch = random.sample(agent.replay_memory, 32)

            next_states = [m[3] for m in minibatch]
            feed_dict = {agent.input: next_states}
            # feed_dict.update(zip(agent.weights, target_weights))
            q_vals = agent.sess.run(agent.output, feed_dict=feed_dict)
            max_q = q_vals.max(axis=1)
            target_q = np.zeros(32)
            action_list = np.zeros((32, agent.OUTPUT_SIZE))
            for i in range(32):
                _, action_index, reward, _, terminal = minibatch[i]
                target_q[i] = reward
                if not terminal:
                    target_q[i] = target_q[i] + 0.99*max_q[i]

                action_list[i][action_index] = 1.0

            states = [m[0] for m in minibatch]
            feed_dict = {agent.input: states, agent.target: target_q,
                         agent.action_hot: action_list}
            _, loss_val = agent.sess.run(fetches=(agent.train_operation, agent.loss), feed_dict=feed_dict)
            loss_vals.append(loss_val)


        if total_steps % 1000 == 0:
            print("updating target weights")
            target_weights = agent.sess.run(agent.weights)

        if total_steps % 3000 == 0:
            save_data(agent, loss_vals, prob, learning_data, weight_average_array, target_weights)
            # testing_thread = Thread(target=save_data, args=(agent, loss_vals, prob,
                                                            # learning_data, weight_average_array))
            # testing_thread.start()

        total_steps += 1
        steps += 1

    episode_number += 1

    episode_step_count.append(steps)
    mean_steps = np.mean(episode_step_count[-100:])
    print("Training episode = {}, Total steps = {}, Last 100 mean steps = {}"
          .format(episode_number, total_steps, mean_steps))
