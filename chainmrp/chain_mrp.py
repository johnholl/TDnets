import numpy as np
from random import choice

class ChainMRP:

    def __init__(self, n=5, k=4, r=1, g=0.99, state=0, prob=1.):
        ## n = number of landmarks
        ## k = space between landmarks
        ## r = reward received in absorbing state
        ## g = discount factor
        ## total number of states = n*k
        ## observations are one hot encodings.

        self.num_landmarks = n
        self.obs_dim = n + 1
        self.num_skips = k
        self.num_states = n*k
        self.final_reward = r
        self.discount = g
        self.prob = prob
        self.state = state
        self.env_time = 0
        self.terminal_state = self.num_states - 1

        self.env_matrix = np.zeros(shape=[self.num_states, self.num_landmarks + 1])
        for i in range(self.num_states):
            if not ((i + 1) % self.num_skips == 0):
                self.env_matrix[i][0] = 1.
            else:
                self.env_matrix[i][int((i+1)/self.num_skips)] = 1.

        self.obs = self.get_obs()

    def reset(self, state=0):
        self.state = state
        self.env_time = 0
        self.obs = self.get_obs()
        return self.obs

    def step(self):
        p = np.random.sample()
        done = False
        reward = 0.
        if p>.0:
            self.state += 1
            self.obs = self.get_obs()
            if self.state == self.terminal_state:
                done = True
                reward = self.final_reward

        return self.obs, reward, done

    def get_obs(self, s=None):
        if s is None:
            obs = self.env_matrix[self.state]
        else:
            obs = self.env_matrix[s]
        return obs

    def get_random_state(self):
        s = choice(range(self.num_states - 1))
        return s

    def get_next_k_obs(self, k):
        obs_list = []
        for i in range(k):
            obs_list.append(self.get_obs(s=self.state+i+1))

        return obs_list

    def get_discounted_sum(self, gamma):
        s = self.state
        sum = np.zeros(shape=self.obs_dim)
        for i in range(s, self.num_states):
            discounted_obs = self.get_obs(s=i)*gamma**(i-s)
            sum += discounted_obs
        return sum





# cmrp = ChainMRP()
# print(cmrp.env_matrix)


class CustomChainMRP:

    def __init__(self, arr, r=1, g=0.99, state=0, prob=1.):
        ## arr is a 1D array of 0s and 1s. A 0 means
        ## that theobservation is indistinguishable
        ## and a 1 means unique state.
        ## r = reward received in absorbing state
        ## g = discount factor
        ## total number of states = length of array
        ## observations are one hot encodings.

        self.num_landmarks = arr.count(1)
        self.obs_dim = self.num_landmarks + 1
        self.num_states = len(arr)
        self.final_reward = r
        self.discount = g
        self.prob = prob
        self.state = state
        self.env_time = 0
        self.terminal_state = self.num_states - 1

        self.env_matrix = np.zeros(shape=[self.num_states, self.num_landmarks + 1])
        hot_index = 1
        for i in range(self.num_states):
            if arr[i] == 0:
                self.env_matrix[i][0] = 1.
            elif arr[i] == 1:
                self.env_matrix[i][hot_index] = 1.
                hot_index += 1

        self.obs = self.get_obs()

    def reset(self, state=0):
        self.state = state
        self.env_time = 0
        self.obs = self.get_obs()
        return self.obs

    def step(self):
        p = np.random.sample()
        done = False
        reward = 0.
        if p>.0:
            self.state += 1
            self.obs = self.get_obs()
            if self.state == self.terminal_state:
                done = True
                reward = self.final_reward

        return self.obs, reward, done

    def get_obs(self, s=None):
        if s is None:
            obs = self.env_matrix[self.state]
        else:
            obs = self.env_matrix[s]
        return obs

    def get_random_state(self):
        s = choice(range(self.num_states - 1))
        return s

    def get_next_k_obs(self, k):
        obs_list = []
        for i in range(k):
            obs_list.append(self.get_obs(s=self.state+i+1))

        return obs_list

    def get_discounted_sum(self):
        s = self.state
        sum = np.zeros(shape=self.obs_dim)
        for i in range(s, self.num_states):
            discounted_obs = self.get_obs(s=i)*self.discount**(i-s)
            sum += discounted_obs
        return sum

    def get_all_discounted_sums(self, gamma):
        s = self.state
        sum = np.zeros(shape=self.obs_dim)
        for i in range(s, self.num_states):
            discounted_obs = self.get_obs(s=i)*gamma**(i-s)
            sum += discounted_obs
        return sum













