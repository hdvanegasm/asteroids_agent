import random

import torch

import constants
import wrappers


def get_fixed_states(agent, env_id):

    fixed_states = []

    env_game = wrappers.make_env(env_id)

    env_game.reset()

    for steps in range(agent.n_test_fixed_states):
        if constants.SHOW_SCREEN:
            env_game.render()

        obs, _, done, _ = env_game.step(env_game.action_space.sample())  # take a random action

        if done:
            obs = env_game.reset()

        obs_tensor = torch.tensor(obs, device=agent.device, dtype=torch.float32)
        fixed_states.append(obs_tensor.detach())

    env_game.close()
    return fixed_states


def test_agent(agent, env_id):
    game_env = wrappers.make_env(env_id)

    steps = 0
    n_episodes = 0
    sum_score = 0
    sum_reward = 0
    sum_score_episode = 0
    sum_reward_episode = 0

    done_last_episode = False

    while steps <= constants.N_TEST_STEPS:

        sum_score_episode = 0
        sum_reward_episode = 0

        state = game_env.reset()
        state = torch.tensor(state, device=agent.device, dtype=torch.float32)
        while steps <= constants.N_TEST_STEPS:
            action = agent.select_test_action(state)
            obs, (reward, score), done, info = game_env.step(action.item())

            sum_score_episode += score
            sum_reward_episode += reward

            if done:
                next_state = None
            else:
                next_state = torch.tensor(obs, device=agent.device, dtype=torch.float32)

            if next_state is not None:
                state = next_state.clone().detach()

            steps += 1
            done_last_episode = done

            if done:
                break

        if done_last_episode:
            sum_score += sum_score_episode
            sum_reward += sum_reward_episode
            n_episodes += 1

    game_env.close()

    if n_episodes == 0:
        n_episodes = 1
        sum_score = sum_score_episode
        sum_reward = sum_reward_episode

    # Compute Q-values
    sum_q_values = 0
    for state in agent.fixed_states:
        with torch.no_grad():
            sum_q_values += agent.target_net(state).max(1)[0].item()

    return sum_reward / n_episodes, sum_score / n_episodes, n_episodes, float(sum_q_values) / len(agent.fixed_states)
