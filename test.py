import constants
from agent import get_screen
import utils

import gym
import torch

import random


def get_fixed_states():

    fixed_states = []

    env = gym.make('Asteroids-v0')

    cumulative_screenshot = []

    def prepare_cumulative_screenshot(cumul_screenshot):
        # Prepare the cumulative screenshot
        padding_image = torch.zeros((1, constants.STATE_IMG_HEIGHT, constants.STATE_IMG_WIDTH))
        for i in range(constants.N_IMAGES_PER_STATE - 1):
            cumul_screenshot.append(padding_image)

        screen_grayscale_state = get_screen(env)
        cumul_screenshot.append(screen_grayscale_state)


    prepare_cumulative_screenshot(cumulative_screenshot)
    env.reset()

    for steps in range(constants.N_STEPS_FIXED_STATES):
        if constants.SHOW_SCREEN:
            env.render()

        _, _, done, _ = env.step(env.action_space.sample())  # take a random action

        if done:
            env.reset()
            cumulative_screenshot = []
            prepare_cumulative_screenshot(cumulative_screenshot)

        screen_grayscale = get_screen(env)
        cumulative_screenshot.append(screen_grayscale)
        cumulative_screenshot.pop(0)
        state = utils.process_state(cumulative_screenshot)

        fixed_states.append(state)

    env.close()
    return fixed_states


def select_action(state, policy_nn, env):
    epsilon_threshold = constants.TEST_EPSILON
    sample = random.random()
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_nn(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], dtype=torch.long)


def test_agent(target_nn, fixed_states):
    env = gym.make('Asteroids-v0')

    steps = 0
    n_episodes = 0
    sum_score = 0
    sum_reward = 0
    sum_score_episode = 0
    sum_reward_episode = 0

    done_last_episode = False

    while steps <= constants.N_TEST_STEPS:
        cumulative_screenshot = []

        sum_score_episode = 0
        sum_reward_episode = 0

        # Prepare the cumulative screenshot
        padding_image = torch.zeros((1, constants.STATE_IMG_HEIGHT, constants.STATE_IMG_WIDTH))
        for i in range(constants.N_IMAGES_PER_STATE - 1):
            cumulative_screenshot.append(padding_image)

        env.reset()

        screen_grayscale_state = get_screen(env)
        cumulative_screenshot.append(screen_grayscale_state)

        state = utils.process_state(cumulative_screenshot)

        prev_state_lives = constants.INITIAL_LIVES

        while steps <= constants.N_TEST_STEPS:
            action = select_action(state, target_nn, env)
            _, reward, done, info = env.step(action)

            sum_score_episode += reward

            reward_tensor = None

            if info["ale.lives"] < prev_state_lives:
                sum_reward_episode += -1
            elif reward > 0:
                sum_reward_episode += 1

            prev_state_lives = info["ale.lives"]

            screen_grayscale = get_screen(env)
            cumulative_screenshot.append(screen_grayscale)
            cumulative_screenshot.pop(0)

            if done:
                next_state = None
            else:
                next_state = utils.process_state(cumulative_screenshot)

            state = next_state
            steps += 1
            done_last_episode = done

            if done:
                break

        if done_last_episode:
            sum_score += sum_score_episode
            sum_reward += sum_reward_episode
            n_episodes += 1

    env.close()

    if n_episodes == 0:
        n_episodes = 1
        sum_score = sum_score_episode
        sum_reward = sum_reward_episode

    # Compute Q-values
    sum_q_values = 0
    for state in fixed_states:
        sum_q_values += target_nn(state).max(1)[1]

    return sum_reward / n_episodes, sum_score / n_episodes, n_episodes, sum_q_values.item() / len(fixed_states)