from model import BehaviorAgent, DeterministicBehaviorAgent
from lab_interface import LabInterface
import numpy as np
import tensorflow as tf

env = LabInterface(level='seekavoid_arena_01')
sess = tf.Session()



beh_policy = DeterministicBehaviorAgent(ob_space=env.observation_space_shape, ac_space=env.num_actions,
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

while True:
    terminal_end = False

    for _ in range(20):
        fetched = beh_policy.act(last_state, prev_action, prev_reward, *last_beh_features, current_session=sess)
        action, beh_features = fetched[0], fetched[2:]
        action_index = np.where(action==1)
        state, reward, terminal = env.step(action_index[0][0])

        # collect the experience
        length += 1
        rewards += reward

        prev_action = action
        prev_reward = reward
        last_state = state
        last_beh_features = beh_features

        if terminal:
            terminal_end = True
            # if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
            last_state = env.reset()
            last_beh_features = beh_policy.get_initial_features()
            print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
            length = 0
            rewards = 0
            break
