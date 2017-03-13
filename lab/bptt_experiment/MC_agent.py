from model import BehaviorAgent, DeterministicBehaviorAgent
from lab_interface import LabInterface, PixLabInterface
import numpy as np
import tensorflow as tf
import pickle
import sys

def save_obj(obj, name ):
    with open('/home/john/objects/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('/home/john/objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

env = PixLabInterface(level='seekavoid_arena_01', num_cuts=20)
sess = tf.Session()

MC_reward_counts = {}
MC_pixel_counts = {}




# beh_policy = DeterministicBehaviorAgent(ob_space=env.observation_space_shape, ac_space=env.num_actions,
#                                    sess=sess,
#                                    ckpt_file="/home/john/tmp/train/model.ckpt-11471320")

beh_policy = BehaviorAgent(ob_space=env.observation_space_shape, ac_space=env.num_actions,
                                   sess=sess,
                                   ckpt_file="/home/john/tmp/train/model.ckpt-11471320")

saver = tf.train.Saver()
beh_policy.restore(sess=sess, saver=saver)

last_state, last_pc = env.reset()
last_beh_features = beh_policy.get_initial_features()
length = 0
rewards = 0
prev_reward = 0
value_ = 0
prev_action = np.zeros(shape=[env.num_actions])
num_steps = 0
episode_observationvals = {}
episode_pixelvals = {}

while True:
    episode_observationvals[str(last_state)] = 0.
    episode_pixelvals[str(last_state)] = np.zeros(shape=[20,20])
    if str(last_state) in MC_reward_counts.keys():
        MC_reward_counts[str(last_state)][0] += 1.
    else:
        MC_reward_counts[str(last_state)] = [1., 0.]

    if str(last_state) in MC_pixel_counts.keys():
        pass
    else:
        MC_pixel_counts[str(last_state)] = np.zeros(shape=[20,20])

    fetched = beh_policy.act(last_state, prev_action, prev_reward, *last_beh_features, current_session=sess)
    action, beh_features = fetched[0], fetched[2:]
    action_index = np.where(action==1)
    state, reward, terminal, pixchange = env.step(action_index[0][0])


    # collect the experience
    length += 1
    rewards += reward
    for epob in episode_observationvals.keys():
        disc_factor = .99**(length-episode_observationvals.keys().index(epob))
        disc_rew = reward * disc_factor
        disc_pixchange = pixchange * disc_factor
        episode_observationvals[epob] += disc_rew
        episode_pixelvals[epob] += disc_pixchange

    prev_action = action
    prev_reward = reward
    last_state = state
    last_beh_features = beh_features


    if terminal:
        for epob in episode_observationvals.keys():
            MC_reward_counts[epob][1] = MC_reward_counts[epob][1] * ((MC_reward_counts[epob][0] - 1) / MC_reward_counts[epob][0]) + episode_observationvals[epob] / MC_reward_counts[epob][0]
            MC_pixel_counts[epob] = MC_pixel_counts[epob] * ((MC_reward_counts[epob][0] - 1) / MC_reward_counts[epob][0]) + episode_pixelvals[epob] / MC_reward_counts[epob][0]

        last_state, last_pc = env.reset()
        last_beh_features = beh_policy.get_initial_features()
        print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
        length = 0
        rewards = 0
        episode_observationvals = {}
        episode_pixelvals = {}

    num_steps += 1
    if num_steps % 100 == 0:
        print(len(MC_reward_counts.keys()))
        print(np.average(MC_reward_counts.values(), axis=0))
        print(np.average(MC_pixel_counts.values()))
        # print(np.average(np.sum(MC_pixel_counts.values(), axis=(1,2))))

        if len(MC_reward_counts.keys()) > 20000:
            # save_obj(MC_reward_counts, name="MC_experiment_20000_det")
            sys.exit()



