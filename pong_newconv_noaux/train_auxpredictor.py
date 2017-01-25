import numpy as np
from agents import FCRecurrentAgent
import random
from processing import preprocess



TARGET_UPDATE_FREQUENCY = 3000
TEST_FREQUENCY = 5000
OBS_DIM = 32
NUM_ACTIONS = 4
PRED_TYPE = "Grid"
PROB = 1.
ANNEAL_RATE = 500000
TOTAL_STEPS = 1000000
MINIBATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99

loss_arr = []

agent = FCRecurrentAgent(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, pred_type=PRED_TYPE, start_prob=PROB, end_prob=.1,
                         anneal_rate=ANNEAL_RATE, checkpoint_name="Auxpredictor_Grid")

target_weights = agent.sess.run(agent.weights)
total_steps = 0.
episode_num = 0.

while total_steps < TOTAL_STEPS:
    obs_sequence = np.zeros(shape=[agent.max_time, OBS_DIM, OBS_DIM, 3])
    reward_sequence = np.zeros(shape=[agent.max_time])
    intensity_sequence = np.zeros(shape=[agent.max_time, agent.predictions.num_cuts, agent.predictions.num_cuts])
    done_sequence = np.zeros(shape=[agent.max_time])
    action_sequence = np.zeros(shape=[agent.max_time])
    obs = preprocess(agent.train_env.reset())
    print(np.shape(obs))
    obs_sequence = np.append(obs_sequence, [obs], axis=0)
    obs_sequence = np.delete(obs_sequence, [0], axis=0)
    new_obs_sequence = obs_sequence
    episode_num += 1
    done = False
    steps=0

    while not done:
        action, reward, term, new_obs, _ = agent.network_step(obs_sequence, agent.train_env)
        new_obs = preprocess(new_obs)
        new_obs_sequence = np.append(new_obs_sequence, [new_obs], axis=0)
        new_obs_sequence = np.delete(new_obs_sequence, [0], axis=0)
        reward_sequence = np.append(reward_sequence, [reward], axis=0)
        intensity_changes = agent.predictions.calculate_intensity_change(obs, new_obs)
        intensity_sequence = np.append(intensity_sequence, [intensity_changes], axis=0)
        action_sequence = np.append(action_sequence, [action], axis=0)
        done_sequence = np.append(done_sequence, [term], axis=0)
        agent.update_replay_memory(tuple=(obs_sequence, reward_sequence, new_obs_sequence, intensity_sequence,
                                          done_sequence, action_sequence))
        obs = new_obs
        obs_sequence = np.append(obs_sequence, [new_obs], axis=0)
        obs_sequence = np.delete(obs_sequence, [0], axis=0)
        steps += 1
        total_steps +=1
        done = term

        if len(agent.replay_memory) > 50000:
            minibatch = random.sample(agent.replay_memory, k=MINIBATCH_SIZE)
            obs_sequences = np.array([m[0] for m in minibatch])
            next_obs_sequences = np.array([m[2] for m in minibatch])
            reward_sequences = np.array([m[1] for m in minibatch])
            intensity_sequences = np.array([m[3] for m in minibatch])
            done_sequences = np.array([m[4] for m in minibatch])
            action_sequences = np.array([m[5] for m in minibatch])

            # Calculate target values for Q and P
            feed_dict = {agent.input: next_obs_sequences, agent.batch_size: MINIBATCH_SIZE}
            feed_dict.update(zip(agent.weights, target_weights))
            (q_vals, predictions) = agent.sess.run((agent.Q_output, agent.P_output), feed_dict=feed_dict)
            max_q = q_vals.max(axis=2)
            max_p = predictions.max(axis=4)
            target_q = np.zeros(shape=[MINIBATCH_SIZE, agent.max_time-agent.masklength])
            target_p = np.zeros(shape=[MINIBATCH_SIZE, agent.max_time-agent.masklength,
                                       agent.predictions.num_cuts, agent.predictions.num_cuts])
            action_list = np.zeros(shape=[MINIBATCH_SIZE, agent.max_time-agent.masklength, agent.num_actions])

            for i in range(MINIBATCH_SIZE):
                for j in range(agent.max_time-agent.masklength):
                    target_q[i][j] = reward_sequence[i][j + agent.masklength]
                    target_p[i][j] = intensity_sequences[i][j+agent.masklength]
                    if not done_sequences[i][j]:
                        target_q[i][j] += max_q[i][j+agent.masklength]
                        target_p[i][j] += max_p[i][j+agent.masklength]

                    action_list[i][j][action_sequences[i][j+agent.masklength]] = 1.



            feed_dict = {agent.input: obs_sequences, agent.Q_target: target_q, agent.P_target: target_p,
                          agent.batch_size: 32, agent.action_hot: action_list}
            _, q_loss, p_loss = agent.sess.run(fetches=(agent.train_operation, agent.Q_loss, agent.P_loss), feed_dict=feed_dict)
            agent.lossarr.append((q_loss, p_loss))

        if total_steps % 1000 == 0:
            print("updating target weights")
            target_weights = agent.sess.run(agent.weights)


        if steps % 10000 == 0:
            # agent.save_data()
            pass


    avg_loss = np.mean(agent.lossarr[-100:])
    print("Most recent average loss: ", avg_loss)
    print(steps)
