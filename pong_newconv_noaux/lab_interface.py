from deepmind_lab import Lab
import numpy as np

class LabInterface():

    def __init__(self, level, observations=[], config={'width': '84', 'height': '84', }):

        self.env = Lab(level=level, observations=observations, config=config)
        self.observation_space_shape = (int(config['height']), int(config['width']), 3)


        # For now, hardcoding number of discrete actions to 8:
        # look left, look right, look up, look down, strafe left, strafe right, forward, backward
        self.num_actions = 8

    def reset(self):
        self.env.reset()
        obs = self.env.observations()['RGB_INTERLACED']
        return obs

    def step(self, action):
        rew = self.env.step(self.convert_int_to_action(action))
        obs = self.env.observations()['RGB_INTERLACED']
        done = not self.env.is_running()
        return obs, rew, done

    def convert_int_to_action(self, index):
        action = np.zeros(shape=[self.num_actions], dtype=np.intc)
        action[index] = 1
        return action
