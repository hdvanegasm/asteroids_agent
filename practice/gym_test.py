import gym



import gym
env = gym.make('BreakoutNoFrameskip-v0')
positive_reward = 0
total_steps = 0
for i_episode in range(50):
    observation = env.reset()
    t = 0
    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if reward != 0:
            positive_reward += 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        t += 1
        total_steps += 1
env.close()

print("# Positive Reward:", positive_reward)
print("# Total Steps:", total_steps)
print("p =", positive_reward / total_steps)