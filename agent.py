import math
import random
import numpy
import pandas

import gym

import torch.optim
import torch.nn.functional

from network import DeepQNetwork
import constants
import memory
import utils
import test
import time

import matplotlib.pyplot as plt


def compute_epsilon(steps_done):
    if steps_done < 1000000:
        return (-9 / 10000000) * steps_done + 1
    else:
        return 0.1


def select_action(state, policy_nn, steps_done, env):
    epsilon_threshold = compute_epsilon(steps_done)
    sample = random.random()
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_nn(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], dtype=torch.long)


def optimize_model(target_nn, policy_nn, memory, optimizer, criterion):
    if len(memory) < constants.BATCH_SIZE:
        return

    transitions = memory.sample(constants.BATCH_SIZE)

    # Array of True/False if the state is not final
    non_final_mask = torch.tensor(tuple(map(lambda t: t.next_state is not None, transitions)))
    non_final_next_states = torch.cat([trans.next_state for trans in transitions
                                       if trans.next_state is not None])
    state_batch = torch.cat([trans.state for trans in transitions])
    action_batch = torch.cat([trans.action for trans in transitions])
    reward_batch = torch.cat([trans.reward for trans in transitions])

    state_action_values = policy_nn(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(constants.BATCH_SIZE)
    next_state_values[non_final_mask] = target_nn(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * constants.GAMMA) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def get_screen(env):
    screen = env.render(mode='rgb_array')
    return (utils.transform_image(screen)[1] * 255).unsqueeze(0)


def plot_loss_continuous(losses):
    plt.figure(2)
    plt.clf()
    plt.title('Training Loss')
    plt.xlabel('Batch Update')
    plt.ylabel('Loss')
    plt.plot(losses)
    # Take 100 episode averages and plot them too

    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_q_continuous(q_values):
    plt.figure(2)
    plt.clf()
    plt.title('Training Q')
    plt.xlabel('Batch Update')
    plt.ylabel('Loss')
    plt.plot(q_values)
    # Take 100 episode averages and plot them too

    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_loss_img(losses):
    plt.figure(2)
    plt.clf()
    plt.title('Training Loss')
    plt.xlabel('Batch Update')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.imsave("losses.png")
    plt.close()


def main_training_loop():
    fixed_states = test.get_fixed_states()

    env = gym.make('AsteroidsNoFrameskip-v0')

    replay_memory = memory.ReplayMemory(constants.REPLAY_MEMORY_SIZE)

    n_actions = env.action_space.n

    policy_net = DeepQNetwork(constants.STATE_IMG_HEIGHT,
                              constants.STATE_IMG_WIDTH,
                              constants.N_IMAGES_PER_STATE // 2,
                              n_actions)

    target_net = DeepQNetwork(constants.STATE_IMG_HEIGHT,
                              constants.STATE_IMG_WIDTH,
                              constants.N_IMAGES_PER_STATE // 2,
                              n_actions)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=constants.LEARNING_RATE, momentum=0.95)

    steps_done = 0
    epoch = 0
    losses = []
    q_values = []
    information = [["epoch", "n_steps", "avg_reward", "avg_score", "n_episodes", "avg_q_value"]]

    try:
        for i_episode in range(constants.N_EPISODES):

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

            for i in range(constants.N_TIMESTEP_PER_EP):
                if constants.SHOW_SCREEN:
                    env.render()

                action = select_action(state, policy_net, steps_done, env)

                _, reward, done, info = env.step(action)
                episode_score += reward

                reward_tensor = None
                if info["ale.lives"] < prev_state_lives:
                    reward_tensor = torch.tensor([-1])
                    episode_reward += -1
                elif reward > 0:
                    reward_tensor = torch.tensor([1])
                    episode_reward += 1
                elif reward < 0:
                    reward_tensor = torch.tensor([-1])
                    episode_reward += -1
                else:
                    reward_tensor = torch.tensor([0])

                prev_state_lives = info["ale.lives"]

                screen_grayscale = get_screen(env)
                cumulative_screenshot.append(screen_grayscale.clone().detach())
                cumulative_screenshot.pop(0)  # Deletes the first element of the list to save memory space

                if done:
                    next_state = None

                    replay_memory.push(state.clone().detach(),
                                       action.clone().detach(),
                                       next_state,
                                       reward_tensor)
                else:
                    next_state = utils.process_state(cumulative_screenshot)

                    replay_memory.push(state.clone().detach(),
                                       action.clone().detach(),
                                       next_state.clone().detach(),
                                       reward_tensor)

                    state = next_state.clone().detach()

                loss = optimize_model(target_net, policy_net, replay_memory, optimizer, criterion)

                if constants.PLOT_LOSS:
                    losses.append(loss)
                    plot_loss_continuous(losses)

                if constants.PLOT_Q:
                    if steps_done > 40:
                        q_values.pop(0)
                    q_values.append(target_net(state).max(1)[0].view(1, 1).item())
                    plot_q_continuous(q_values)

                steps_done += 1

                if done:
                    print("Episode:", i_episode, "Steps done:", steps_done, "- Episode reward:", episode_reward,
                          "- Episode score:", episode_score)
                    break

                # Update target policy
                if steps_done % constants.TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # Epoch test
                if steps_done % constants.STEPS_PER_EPOCH == 0:
                    epoch += 1
                    epoch_reward_average, epoch_score_average, n_episodes, q_values_average = test.test_agent(
                        target_net, fixed_states)
                    information.append(
                        [epoch, steps_done, epoch_reward_average, epoch_score_average, n_episodes, q_values_average])
                    print("INFO:",
                          [epoch, steps_done, epoch_reward_average, epoch_score_average, n_episodes, q_values_average],
                          "Epsilon =", compute_epsilon(steps_done))

                # Save file periodically
                if steps_done % constants.PERIODIC_SAVE == 0:
                    print("Saving network state")
                    torch.save(target_net, "info/nn_parameters.ptf")

        # Save test information in dataframe
        print("Saving information...")
        information_numpy = numpy.array(information)
        dataframe_information = pandas.DataFrame(columns=information_numpy[0, 0:],
                                                 data=information_numpy[1:, 0:])
        dataframe_information.to_csv("info/results.csv")
        print(dataframe_information)


    except KeyboardInterrupt:
        # Save test information in dataframe
        print("Saving information...")
        information_numpy = numpy.array(information)
        dataframe_information = pandas.DataFrame(columns=information_numpy[0, 0:], data=information_numpy[1:, 0:])
        dataframe_information.to_csv("info/results.csv")
        print(dataframe_information)

    env.close()


if __name__ == "__main__":
    main_training_loop()
