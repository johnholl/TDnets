from model import BehaviorAgent, DeterministicBehaviorAgent
from lab_interface import LabInterface
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

env = LabInterface(level='seekavoid_arena_01')
sess = tf.Session()

MC_counts = {}



#beh_policy = DeterministicBehaviorAgent(ob_space=env.observation_space_shape, ac_space=env.num_actions,
#                                    sess=sess,
#                                    ckpt_file="/home/john/tmp/train/model.ckpt-11471320")

beh_policy = BehaviorAgent(ob_space=env.observation_space_shape, ac_space=env.num_actions,
                                   sess=sess,
                                   ckpt_file="/home/john/tmp/train/model.ckpt-11471320")

saver = tf.train.Saver()
beh_policy.restore(sess=sess, saver=saver)

last_state = env.reset()
last_beh_features = beh_policy.get_initial_features()
length = 0
rewards = 0
prev_reward = 0
value_ = 0
prev_action = np.zeros(shape=[env.num_actions])
num_steps = 0
episode_observationvals = {}

while True:
    episode_observationvals[str(last_state)] = 0.
    if str(last_state) in MC_counts.keys():
        MC_counts[str(last_state)][0] += 1.
    else:
        MC_counts[str(last_state)] = [1., 0.]

    fetched = beh_policy.act(last_state, prev_action, prev_reward, *last_beh_features, current_session=sess)
    action, beh_features = fetched[0], fetched[2:]
    action_index = np.where(action==1)
    state, reward, terminal = env.step(action_index[0][0])


    # collect the experience
    length += 1
    rewards += reward
    if reward != 0.:
        for epob in episode_observationvals.keys():
            disc_rew = reward * .99**(length-episode_observationvals.keys().index(epob))
            episode_observationvals[epob] += disc_rew


    prev_action = action
    prev_reward = reward
    last_state = state
    last_beh_features = beh_features


    if terminal:
        for epob in episode_observationvals.keys():
            MC_counts[epob][1] = MC_counts[epob][1]*((MC_counts[epob][0]-1)/MC_counts[epob][0]) + episode_observationvals[epob]/MC_counts[epob][0]


        last_state = env.reset()
        last_beh_features = beh_policy.get_initial_features()
        print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
        length = 0
        rewards = 0
        episode_observationvals = {}

    num_steps += 1
    if num_steps % 100 == 0:
        print(len(MC_counts.keys()))
        print(np.average(MC_counts.values(), axis=0))

        if len(MC_counts.keys()) > 2000000:
            save_obj(MC_counts, name="MC_experiment_2000000")
            sys.exit()



