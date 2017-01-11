import gym

env = gym.make('Taxi-v1')
obs = env.reset()
done = False
while not done:
    env.render()
    a = input("Enter an action number (0-5): ")
    try:
        obs, rew, done, _ = env.step(int(a))
        print(obs)

    except (KeyError, ValueError):
        print("Action not recognized. Try again.")
        pass
