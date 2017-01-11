from chainmrp.chain_mrp import ChainMRP


env = ChainMRP()

print(env.env_matrix)


for i in range(3):
    obs = env.reset()
    print(obs)
    done = False
    while not done:
        a = input("press Enter to continue: ")
        obs, reward, done = env.step()
        print(obs)
        print(reward)
        print(done)
