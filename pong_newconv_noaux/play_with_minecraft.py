import gym
import numpy as np

from processing import preprocess

env = gym.make("MinecraftEating1-v0")
env.init()
env.reset()

obs, action, reward, done = env.step(1)
print(np.shape(obs))
processed_obs = preprocess(obs)
print(np.shape(processed_obs))


