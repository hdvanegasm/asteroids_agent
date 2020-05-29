import random

import gym
import matplotlib.pyplot as plt
import numpy
import pandas
import torch.nn.functional
import torch.optim

import constants
import memory
import test
import utils
from network import DeepQNetwork

# Set device if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)


def optimize_model(target_nn, policy_nn, memory, optimizer, criterion):
    if len(memory) < constants.BATCH_SIZE:
        return

    transitions = memory.sample(constants.BATCH_SIZE)

    # Array of True/False if the state is non final
    non_final_mask = torch.tensor(tuple(map(lambda t: t.next_state is not None, transitions)), dtype=torch.bool,
                                  device=device)
    non_final_next_states = torch.cat([trans.next_state for trans in transitions
                                       if trans.next_state is not None])
    state_batch = torch.cat([trans.state for trans in transitions])
    action_batch = torch.cat([trans.action for trans in transitions])
    reward_batch = torch.cat([trans.reward for trans in transitions])

    state_action_values = policy_nn(state_batch).gather(1, action_batch)

    utils.state_to_image(state_batch[0].unsqueeze(0), "StateA")
    utils.state_to_image(non_final_next_states[0].unsqueeze(0), "StateB")
    import sys
    sys.exit(0)

    next_state_values = torch.zeros(constants.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_nn(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * constants.GAMMA) + reward_batch
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    #  No clamp (origen del error)
    optimizer.step()

    return loss.item()


def get_screen(env):
    screen = env.render(mode='rgb_array')
    return (utils.transform_image(screen)[1]).unsqueeze(0).to(device)  # No escalado (origen del error)


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
    plt.title('Q-Value')
    plt.xlabel('Batch Update')
    plt.ylabel('Q-Value')
    plt.plot(q_values)

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


def plot_scores(scores):
    plt.figure(2)
    plt.clf()
    plt.title("Training scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.savefig("scores.png")
    plt.close()


def main_training_loop():
    fixed_states = test.get_fixed_states()

    env = gym.make('AsteroidsNoFrameskip-v0')

    # Initialize replay memory
    replay_memory = memory.ReplayMemory(constants.REPLAY_MEMORY_SIZE)

    # Configure CNN
    n_actions = env.action_space.n
    policy_net = DeepQNetwork(constants.STATE_IMG_HEIGHT,
                              constants.STATE_IMG_WIDTH,
                              constants.N_IMAGES_PER_STATE // 2,
                              n_actions).to(device)
    target_net = DeepQNetwork(constants.STATE_IMG_HEIGHT,
                              constants.STATE_IMG_WIDTH,
                              constants.N_IMAGES_PER_STATE // 2,
                              n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=constants.LEARNING_RATE) # Momentum (origen del error)

    # Initialize storage data structures
    steps_done = 0
    epoch = 0
    losses = []
    q_values = []
    information = [["epoch", "n_steps", "avg_reward", "avg_score", "n_episodes", "avg_q_value"]]
    episode_scores = []

    try:
        for i_episode in range(constants.N_EPISODES):

            cumulative_screenshot = []

            # Prepare the cumulative screenshot with initial black screens
            for i in range(constants.N_IMAGES_PER_STATE - 1):
                padding_image = torch.zeros((1, constants.STATE_IMG_HEIGHT, constants.STATE_IMG_WIDTH)).detach().to(device)
                cumulative_screenshot.append(padding_image)

            env.reset()
            episode_score = 0
            episode_reward = 0

            # Process and get the first state
            screen_grayscale_state = get_screen(env)
            cumulative_screenshot.append(screen_grayscale_state.clone().detach())
            state = utils.process_state(cumulative_screenshot)

            # Initialize lives
            prev_state_lives = constants.INITIAL_LIVES

            while True:
                if constants.SHOW_SCREEN:
                    env.render()

                # Select an action with epsilon-greedy policy
                action = select_action(state, policy_net, steps_done, env)

                observation_reward = 0

                # Make a step in the environment
                _, reward, done, info = env.step(action.item())
                episode_score += reward

                # Apply transformation on the reward
                if info["ale.lives"] < prev_state_lives:
                    observation_reward = -1
                    episode_reward += -1
                elif reward > 0:
                    observation_reward = 1
                    episode_reward += observation_reward
                elif reward < 0:
                    observation_reward = -1
                    episode_reward += -1

                # Update lives
                prev_state_lives = info["ale.lives"]

                # Add current screen to cumulative screenshots
                screen_grayscale = get_screen(env)
                cumulative_screenshot.append(screen_grayscale.clone().detach())
                cumulative_screenshot.pop(0)  # Deletes the first element of the list to save memory space

                # Add tuple to replay memory
                if done:
                    next_state = None
                    replay_memory.push(state.clone().detach(),
                                       action.clone(),
                                       next_state,
                                       torch.tensor([observation_reward], device=device))
                else:
                    next_state = utils.process_state(cumulative_screenshot)
                    replay_memory.push(state.clone().detach(),
                                       action.clone(),
                                       next_state.clone().detach(),
                                       torch.tensor([observation_reward], device=device))
                    state = next_state.clone().detach()

                # Make an optimization step
                if len(replay_memory) >= constants.INITIAL_REPLAY_MEMORY_SIZE:
                    loss = optimize_model(target_net, policy_net, replay_memory, optimizer, criterion)
                else:
                    print("Filling replay memory...")
                    steps_done = 0

                if constants.PLOT_LOSS:
                    losses.append(loss)
                    plot_loss_continuous(losses)

                if constants.PLOT_Q:
                    if steps_done > 40:
                        q_values.pop(0)

                    with torch.no_grad():  # Use torch.no_grad() for
                        q_values.append(target_net(state).max(1)[0].view(1, 1).item())

                    plot_q_continuous(q_values)

                steps_done += 1

                if steps_done % 300 == 0:
                    with torch.no_grad():
                        print("Epoch:", epoch, "- Q-Value:", target_net(state).max(1)[0].view(1, 1).item(), "- Loss:", loss)

                if done:
                    print("Episode:", i_episode, "- Steps done:", steps_done, "- Episode reward:", episode_reward,
                          "- Episode score:", episode_score)
                    episode_scores.append(episode_score)
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
                    print("Saving network state...")
                    torch.save(target_net.state_dict(), "info/nn_parameters.pth")
                    print("Network state saved.")
                    print("Saving information...")
                    information_numpy = numpy.array(information)
                    dataframe_information = pandas.DataFrame(columns=information_numpy[0, 0:],
                                                             data=information_numpy[1:, 0:])
                    dataframe_information.to_csv("info/results.csv")
                    print(dataframe_information)
                    plot_scores(episode_scores)
                    print("Information saved")

        # Save test information in dataframe
        print("Saving network state...")
        torch.save(target_net.state_dict(), "info/nn_parameters.pth")
        print("Network state saved.")

        print("Saving information...")
        information_numpy = numpy.array(information)
        dataframe_information = pandas.DataFrame(columns=information_numpy[0, 0:],
                                                 data=information_numpy[1:, 0:])
        dataframe_information.to_csv("info/results.csv")
        print(dataframe_information)

        plot_scores(episode_scores)

        print("Information saved")

    except KeyboardInterrupt:
        # Save test information in dataframe
        print("Saving network state...")
        torch.save(target_net.state_dict(), "info/nn_parameters.pth")
        print("Network state saved.")

        print("Saving information...")
        information_numpy = numpy.array(information)
        dataframe_information = pandas.DataFrame(columns=information_numpy[0, 0:], data=information_numpy[1:, 0:])
        dataframe_information.to_csv("info/results.csv")
        print(dataframe_information)

        plot_scores(episode_scores)

        print("Information saved")

    env.close()


if __name__ == "__main__":
    main_training_loop()
