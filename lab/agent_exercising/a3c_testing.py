from model import AuxLSTMPolicy
import deepmind_lab

env = deepmind_lab.Lab('seekavoid_arena_01', observations=['RGB_INTERLACED'])
policy = AuxLSTMPolicy()

last_state = env.reset()
last_features = policy.get_initial_features()
length = 0
rewards = 0
prev_reward = 0
prev_action = np.zeros(shape=[env.num_actions])

while True:
    terminal_end = False
    rollout = PartialRollout()

    for _ in range(num_local_steps):
        # print(np.shape([[prev_reward]]))
        # print(np.shape(last_features[0]))
        # print(np.shape(last_features[1]))
        fetched = policy.act(last_state, prev_action, prev_reward, *last_features)
        action, value_, features = fetched[0], fetched[1], fetched[2:]
        # argmax to convert from one-hot
        state, reward, terminal = env.step(action.argmax())

        # collect the experience
        rollout.add(last_state, action, reward, value_, terminal, last_features, prev_action, [prev_reward])
        length += 1
        rewards += reward

        prev_action = action
        prev_reward = reward
        last_state = state
        last_features = features

