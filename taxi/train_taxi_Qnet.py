import numpy as np
from taxi.taxi_TDnet import TaxiQNet
import random
from threading import Thread
import sys


TARGET_UPDATE_FREQUENCY = 3000
TEST_FREQUENCY = 5000
CONV = False
PROB = 1.
ANNEAL_RATE = 5000
TOTAL_STEPS = 1000000
MINIBATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99

loss_arr = []

agent = TaxiQNet()

target_weights = agent.sess.run(agent.weights)
total_steps = 0.
episode_num = 0.

while total_steps < TOTAL_STEPS:
    obs = agent.train_env.reset()
    steps = 0
    done = False
    while not done:
        action, reward, term, new_obs, _ = agent.network_step(obs, env=agent.train_env)
        agent.update_replay_memory((obs, action, reward, new_obs, term))
        steps += 1
        total_steps += 1
        done = term

        if total_steps > agent.replay_size:                 # currently does a training update every step
            minibatch = random.sample(agent.replay_memory, MINIBATCH_SIZE)
            # agent.learn()
            next_states = np.array([m[3] for m in minibatch])
            formatted_next_states = np.zeros(shape=[MINIBATCH_SIZE, 500])
            for i in range(MINIBATCH_SIZE):
                formatted_next_states[i] = agent.format_to_standard(next_states[i])
            feed_dict = {agent.input: formatted_next_states}
            feed_dict.update(zip(agent.weights, target_weights))
            q_vals = agent.sess.run(agent.output, feed_dict=feed_dict)
            max_q = q_vals.max(axis=1)
            target_q = np.zeros(32)
            action_list = np.zeros((MINIBATCH_SIZE, agent.output_size))
            for i in range(MINIBATCH_SIZE):
                _, action_index, reward, _, terminal = minibatch[i]
                target_q[i] = reward
                if not terminal:
                    target_q[i] += DISCOUNT_FACTOR*max_q[i]

                action_list[i][action_index] = 1.0

            states = [m[0] for m in minibatch]
            formatted_states = np.zeros(shape=[MINIBATCH_SIZE, 500])
            for i in range(MINIBATCH_SIZE):
                formatted_states[i] = agent.format_to_standard(states[i])
            feed_dict = {agent.input: np.array(formatted_states), agent.target: target_q, agent.action_hot: action_list}
            _, loss_value = agent.sess.run((agent.train_operation, agent.loss), feed_dict=feed_dict)
            agent.loss_arr.append(loss_value)


        if total_steps % TARGET_UPDATE_FREQUENCY == 0:
            print("updating target weights")
            target_weights = agent.sess.run(agent.weights)

        if total_steps % TEST_FREQUENCY == 0:
            # testing_thread = Thread(target=agent.save_data)
            # testing_thread.start()
            pass

        if total_steps % 500 == 0:
            print("Total steps = {}".format(total_steps))
