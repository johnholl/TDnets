from deepmind_lab import Lab
import numpy as np
import time

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
        obs = self.env.observations()['RGB_INTERLACED']
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


