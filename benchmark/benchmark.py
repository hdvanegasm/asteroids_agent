from network import DeepQNetwork
from agent import get_screen
from agent import plot_q_continuous
import constants
import utils
import time

import torch

import gym
import random
import numpy


def select_action(state, policy_nn, env):
    epsilon_threshold = constants.TEST_EPSILON
    sample = random.random()
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_nn(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], dtype=torch.long)


def benchmark():
    env = gym.make('AsteroidsNoFrameskip-v0')

    n_actions = env.action_space.n

    target_net = DeepQNetwork(constants.STATE_IMG_HEIGHT,
                              constants.STATE_IMG_WIDTH,
                              constants.N_IMAGES_PER_STATE // 2,
                              n_actions)

    target_net.load_state_dict(torch.load("nn_parameters.pth"))

    target_net.eval()

    n_test_episodes = 50

    episode_scores = []
    episode_rewards = []

    steps_done = 0
    q_values = []

    try:
        for i_episode in range(n_test_episodes):

            cumulative_screenshot = []

            # Prepare the cumulative screenshot

            for i in range(constants.N_IMAGES_PER_STATE - 1):
                padding_image = torch.zeros((1, constants.STATE_IMG_HEIGHT, constants.STATE_IMG_WIDTH))
                cumulative_screenshot.append(padding_image)

            env.reset()
            episode_score = 0
            episode_reward = 0

            screen_grayscale_state = get_screen(env)
            cumulative_screenshot.append(screen_grayscale_state.clone().detach())

            state = utils.process_state(cumulative_screenshot)

            prev_state_lives = constants.INITIAL_LIVES

            while True:

                time.sleep(0.01)

                if constants.SHOW_SCREEN:
                    env.render()

                action = select_action(state, target_net, env)
                _, reward, done, info = env.step(action.item())
                episode_score += reward

                if info["ale.lives"] < prev_state_lives:
                    episode_reward += -1
                elif reward > 0:
                    episode_reward += 1
                elif reward < 0:
                    episode_reward += -1

                prev_state_lives = info["ale.lives"]

                screen_grayscale = get_screen(env)
                cumulative_screenshot.append(screen_grayscale.clone().detach())
                cumulative_screenshot.pop(0)  # Deletes the first element of the list to save memory space

                if done:
                    next_state = None
                else:
                    next_state = utils.process_state(cumulative_screenshot)

                if next_state is not None:
                    state = next_state.clone().detach()

                if constants.PLOT_Q:
                    if steps_done > 120:
                        q_values.pop(0)

                    with torch.no_grad():
                        q_values.append(target_net(state).max(1)[0].view(1, 1).item())

                    plot_q_continuous(q_values)

                if done:
                    print("Episode:", i_episode, "- Episode reward:", episode_reward, "- Episode score:", episode_score)
                    episode_scores.append(episode_score)
                    episode_rewards.append(episode_reward)
                    break

                steps_done += 1

        print("RESULTS:")
        print("Score mean:", numpy.mean(episode_scores))
        print("Score standard deviation:", numpy.std(episode_scores))
        print("Reward mean:", numpy.mean(episode_rewards))
        print("Reward standard deviation:", numpy.std(episode_rewards))

    except KeyboardInterrupt:
        print("RESULTS:")
        print("Score mean:", numpy.mean(episode_scores))
        print("Score standard deviation:", numpy.std(episode_scores))
        print("Reward mean:", numpy.mean(episode_rewards))
        print("Reward standard deviation:", numpy.std(episode_rewards))

    env.close()


if __name__ == "__main__":
    benchmark()
