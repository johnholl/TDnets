from deepmind_lab import Lab
import numpy as np
from pixel_helpers import calculate_intensity_change

class LabInterface():

    def __init__(self, level, observations=['RGB_INTERLACED'], config={'width': '84', 'height': '84', }):

        self.env = Lab(level=level, observations=observations, config=config)
        self.observation_space_shape = (int(config['height']), int(config['width']), 3)


        # For now, hardcoding number of discrete actions to 8:
        # look left, look right, look up, look down, strafe left, strafe right, forward, backward
        self.obs = np.zeros(shape=[84,84,3])
        print("interface built")

        self.ACTIONS = [self._action(-20, 0, 0, 0, 0, 0, 0),
          self._action(20, 0, 0, 0, 0, 0, 0),
          self._action(0, 10, 0, 0, 0, 0, 0),
          self._action(0, -10, 0, 0, 0, 0, 0),
          self._action(0, 0, -1, 0, 0, 0, 0),
          self._action(0, 0, 1, 0, 0, 0, 0),
          self._action(0, 0, 0, 1, 0, 0, 0),
          self._action(0, 0, 0, -1, 0, 0, 0)]
          # self._action(0, 0, 0, 0, 1, 0, 0),
          # self._action(0, 0, 0, 0, 0, 1, 0),
          # self._action(0, 0, 0, 0, 0, 0, 1)]

        self.num_actions = len(self.ACTIONS)



    def reset(self):
        self.env.reset()
        obs = self.env.observations()['RGB_INTERLACED']/255.
        self.obs = obs
        return obs

    def step(self, action):
        rew = self.env.step(self.convert_int_to_action(action), num_steps=4)
        if self.env.is_running():
            self.obs = self.env.observations()['RGB_INTERLACED']/255.
        done = not self.env.is_running()

        return self.obs, rew, done

    def convert_int_to_action(self, index):
        action = self.ACTIONS[index]
        return action

    def _action(self, *entries):
        return np.array(entries, dtype=np.intc)


class PixLabInterface(LabInterface):
    def __init__(self, num_cuts, level, observations=['RGB_INTERLACED'], config={'width': '84', 'height': '84', }):
        LabInterface.__init__(self, level=level, observations=observations, config=config)
        self.prev_obs = np.zeros(shape=[84,84,3])
        self.obs = np.zeros(shape=[84,84,3])
        self.num_cuts = num_cuts

    def reset(self):
        self.env.reset()
        obs = self.env.observations()['RGB_INTERLACED']/255.
        self.prev_obs = np.zeros(shape=[84,84,3])
        self.obs = obs
        pix_change = self.compute_pix_change()
        return obs, pix_change

    def step(self, action):
        rew = self.env.step(self.convert_int_to_action(action), num_steps=4)
        if self.env.is_running():
            self.prev_obs = self.obs
            self.obs = self.env.observations()['RGB_INTERLACED']/255.
        done = not self.env.is_running()
        pix_change = self.compute_pix_change()

        return self.obs, rew, done, pix_change

    def compute_pix_change(self):
        pix_change = calculate_intensity_change(self.prev_obs, self.obs, num_cuts=self.num_cuts)
        return pix_change

