import numpy as np
import time
import os
import random
import sys
from chainmrp.chain_mrp import ChainMRP
from chainmrp.value_iterator import ValueIterator

experiment_time = str(time.time())
dir = os.path.dirname(__file__)
save_path = os.path.join(dir, "checkpoints", "iteratormodel_" + experiment_time + ".ckpt")

env = ChainMRP()
valueNet = ValueIterator(obs_dim=env.obs_dim, max_time=4)
gamma = 0.99
out=0
masklength = int(valueNet.max_time/2)

steps = 0
loss_vals = []
episode_num = 0

while steps < 100000:
    obs_sequence = np.zeros(shape=[valueNet.max_time, valueNet.input_size])
    obs = env.reset()
    obs_sequence = np.append(obs_sequence, [obs], axis=0)
    obs_sequence = np.delete(obs_sequence, [0], axis=0)
    episode_num += 1
    done = False

    while not done:
        new_obs, reward, done = env.step()
        valueNet.update_replay_memory(example=(obs, reward, new_obs, done))
        obs = new_obs
        obs_sequence = np.append(obs_sequence, [obs], axis=0)
        obs_sequence = np.delete(obs_sequence, [0], axis=0)
        steps +=1

        if len(valueNet.replay_memory) > 200:
            minibatch = []
            for _ in range(32):
                buf = []
                starting_point = random.choice(range(len(valueNet.replay_memory) - valueNet.max_time))
                while(len(buf) < valueNet.max_time):
                    for j in range(starting_point, starting_point + valueNet.max_time):
                        buf.append(valueNet.replay_memory[j])
                        if valueNet.replay_memory[j][3] and len(buf)<valueNet.max_time:
                            buf = []
                            starting_point = random.choice(range(len(valueNet.replay_memory) - valueNet.max_time))
                            break

                minibatch.append(buf)

            next_obs = [[s[2] for s in m] for m in minibatch]
            rewards = np.expand_dims([[s[1] for s in m] for m in minibatch], axis=2)

            feed_dict = {valueNet.input: next_obs, valueNet.batch_size: 32}
            next_values = valueNet.sess.run(valueNet.output, feed_dict=feed_dict)


            value_targets = np.add(gamma*next_values, rewards)[:, masklength:, :]


            states = [[s[0] for s in m] for m in minibatch]

            feed_dict = {valueNet.input: np.array(states), valueNet.targets: value_targets,
                         valueNet.batch_size: 32}
            _, loss_val, out = valueNet.sess.run(fetches=(valueNet.train_operation, valueNet.loss, valueNet.output), feed_dict=feed_dict)
            loss_vals.append(loss_val)



    avg_loss = np.mean(loss_vals[-100:])
    print("Most recent average loss: ", avg_loss)
    print(steps)
    # np.save(os.path.join(dir, "loss_" + experiment_time), loss_vals)
    # valueNet.save_data(path=save_path)