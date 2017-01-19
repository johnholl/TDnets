import gym
import random

env = gym.make('Taxi-v1')
obs = env.reset()
done = False
step=0

while not done:
    # env.render()
    # a = input("Enter an action number (0-5): ")
    try:
        # obs, rew, done, _ = env.step(int(a))
        env.step(random.choice([0,1,2,3,4,5]))
        # print(obs)
        step +=1
        print(step)

    except (KeyError, ValueError):
        print("Action not recognized. Try again.")
        pass

print(step)
