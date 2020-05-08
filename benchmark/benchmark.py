from network import DeepQNetwork
import constants
import utils

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


def get_screen(env):
    screen = env.render(mode='rgb_array')
    return utils.transform_image(screen)


def benchmark():
    env = gym.make('AsteroidsNoFrameskip-v0')

    n_actions = env.action_space.n

    target_net = DeepQNetwork(constants.STATE_IMG_HEIGHT,
                              constants.STATE_IMG_WIDTH,
                              constants.N_IMAGES_PER_STATE,
                              n_actions)

    n_test_episodes = 1000

    episode_scores = []
    episode_rewards = []

    try:
        for i_episode in range(n_test_episodes):

            cumulative_screenshot = []

            # Prepare the cumulative screenshot
            padding_image = torch.zeros((1, constants.STATE_IMG_HEIGHT, constants.STATE_IMG_WIDTH))
            for i in range(constants.N_IMAGES_PER_STATE - 1):
                cumulative_screenshot.append(padding_image)

            env.reset()
            episode_score = 0
            episode_reward = 0

            screen_grayscale_state = get_screen(env)
            cumulative_screenshot.append(screen_grayscale_state)

            state = utils.process_state(cumulative_screenshot)

            prev_state_lives = constants.INITIAL_LIVES

            while True:
                if constants.SHOW_SCREEN:
                    env.render()

                action = select_action(state, target_net, env)
                _, reward, done, info = env.step(action)
                episode_score += reward

                if info["ale.lives"] < prev_state_lives:
                    episode_reward += -1
                elif reward > 0:
                    episode_reward += 1
                elif reward < 0:
                    episode_reward += -1

                prev_state_lives = info["ale.lives"]

                screen_grayscale = get_screen(env)
                cumulative_screenshot.append(screen_grayscale)
                cumulative_screenshot.pop(0)  # Deletes the first element of the list to save memory space

                if done:
                    next_state = None
                else:
                    next_state = utils.process_state(cumulative_screenshot)

                if next_state is not None:
                    state.copy_(next_state)

                if done:
                    print("Episode:", i_episode, "- Episode reward:", episode_reward, "- Episode score:", episode_score)
                    episode_scores.append(episode_score)
                    episode_rewards.append(episode_reward)
                    break

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
