import gym
import utils
import matplotlib

env = gym.make('AsteroidsNoFrameskip-v0')
steps_done = 0
action = 3
for i_episode in range(1):
    observation = env.reset()
    t = 0
    while True:
        action = env.action_space.sample()
        _, reward, done, info = env.step(action)
        screen = env.render(mode='rgb_array')
        matplotlib.pyplot.imsave("screen" + str(steps_done) + ".png", screen)
        steps_done += 1
        if steps_done == 100:
            break

env.close()

