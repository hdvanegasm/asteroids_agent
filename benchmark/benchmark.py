import random
import time

import numpy
import torch

import constants
import wrappers
from main_dqn import plot_q_continuous
from network import DeepQNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(state, policy_nn, env):
    epsilon_threshold = constants.TEST_EPSILON
    sample = random.random()
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_nn(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], dtype=torch.long, device=device)


def benchmark():
    env = wrappers.make_env("BreakoutNoFrameskip-v0")

    n_actions = env.action_space.n

    target_net = DeepQNetwork(constants.STATE_IMG_HEIGHT,
                              constants.STATE_IMG_WIDTH,
                              constants.N_IMAGES_PER_STATE,
                              n_actions).to(device)

    target_net.load_state_dict(torch.load("nn_parameters.pth"))

    target_net.eval()

    n_test_episodes = 200

    episode_scores = []
    episode_rewards = []

    steps_done = 0
    q_values = []

    try:
        for i_episode in range(n_test_episodes):
            state = env.reset()
            state = torch.tensor(state, device=device, dtype=torch.float32)
            episode_score = 0
            episode_reward = 0

            while True:
                time.sleep(0.01)

                if constants.SHOW_SCREEN:
                    env.unwrapped.render()

                action = select_action(state, target_net, env)
                obs, (reward, score), done, info = env.step(action.item())
                episode_score += score

                episode_reward += reward

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(obs, device=device, dtype=torch.float32)

                if next_state is not None:
                    state = next_state.clone().detach()

                if constants.PLOT_Q:
                    # if steps_done > 120:
                    #     q_values.pop(0)

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
