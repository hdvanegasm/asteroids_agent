import gym
import utils
import matplotlib
import time
import wrappers
import numpy
#env = gym.make('BreakoutNoFrameskip-v0')
env = wrappers.make_env("BreakoutNoFrameskip-v0")
steps_done = 0
print(env.action_space.n)
episode_scores = []
for i_episode in range(500):
    observation = env.reset()
    t = 0
    sum_episode = 0
    while True:
        _, (reward, score), done, info = env.step(env.action_space.sample())
        steps_done += 1
        sum_episode += score
        if done:
            print(sum_episode)
            episode_scores.append(sum_episode)
            break

env.close()
print("Mean:", numpy.mean(episode_scores))
print("Std:", numpy.std(episode_scores))

