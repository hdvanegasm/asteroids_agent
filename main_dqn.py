import matplotlib.pyplot as plt
import numpy
import pandas
import torch.nn.functional
import torch.optim

import utils
import constants
import test
import wrappers
import agent_dqn


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
    plt.figure(1)
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


def plot_scores(scores):
    plt.figure(3)
    plt.clf()
    plt.title("Training scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.savefig("scores.png")
    plt.close()


def main_training_loop():
    # Create agent
    env = wrappers.make_env("BreakoutNoFrameskip-v0")

    agent = agent_dqn.Agent(env=env, lr=constants.LEARNING_RATE, alpha=constants.ALPHA, momentum=constants.MOMENTUM,
                            eps=constants.EPS_RMSPROP, batch_size=constants.BATCH_SIZE,
                            target_update=constants.TARGET_UPDATE, initial_eps=constants.EPS_START,
                            final_eps=constants.EPS_END, final_time=constants.FINAL_STEP_EPS,
                            initial_replay_memory_size=constants.INITIAL_REPLAY_MEMORY_SIZE,
                            replay_memory_size=constants.REPLAY_MEMORY_SIZE, periodic_save=constants.PERIODIC_SAVE,
                            state_width=constants.STATE_IMG_WIDTH, state_height=constants.STATE_IMG_HEIGHT,
                            frames_per_state=constants.N_IMAGES_PER_STATE, test_epsilon=constants.TEST_EPSILON,
                            n_test_fixed_states=constants.N_STEPS_FIXED_STATES, gamma=constants.GAMMA)

    epoch = 0
    losses = []
    q_values = []
    information = [["epoch", "n_steps", "avg_reward", "avg_score", "n_episodes", "avg_q_value"]]
    episode_scores = []

    try:
        for i_episode in range(constants.N_EPISODES):
            state = env.reset()
            state = torch.tensor(state, device=agent.device, dtype=torch.float32)

            episode_score = 0
            episode_reward = 0

            while True:
                if constants.SHOW_SCREEN:
                    env.render()
                # Select an action with epsilon-greedy policy
                action = agent.select_action(state)

                # Make a step in the environment
                observation, (reward, score), done, info = env.step(action.item())

                # Apply transformation on the reward
                episode_reward += reward
                episode_score += score

                # Add tuple to replay memory
                if done:
                    next_state = None
                    agent.replay_memory.push(state.detach(),
                                             action,
                                             next_state,
                                             torch.tensor([reward], device=agent.device))
                else:
                    next_state = torch.tensor(observation, device=agent.device, dtype=torch.float32)
                    agent.replay_memory.push(state.detach(),
                                             action,
                                             next_state.detach(),
                                             torch.tensor([reward], device=agent.device))
                    state = next_state.detach()

                # Make an optimization step. If there is no enough elements to populate replay memory, the steps are
                # restarted and the agent waits for more elements in the replay memory
                loss = None
                if len(agent.replay_memory) >= agent.replay_memory.initial_replay_memory_size:
                    loss = agent.optimize()

                if len(agent.replay_memory) == agent.replay_memory.initial_replay_memory_size:
                    print("Replay memory filled")

                if constants.PLOT_LOSS and agent.steps_done > 0:
                    losses.append(loss)
                    plot_loss_continuous(losses)

                if constants.PLOT_Q and agent.steps_done > 0:
                    with torch.no_grad():
                        q_values.append(agent.target_net(state).max(1)[0].view(1, 1).item())

                    if len(q_values) > 40:
                        q_values.pop(0)

                    plot_q_continuous(q_values)

                if agent.steps_done % 300 == 0 and agent.steps_done > 0:
                    with torch.no_grad():
                        print("Memory:", agent.replay_memory.get_allocated_memory())
                        print("Epoch:", epoch, " - Steps:", agent.steps_done, "- Q-Value:",
                              agent.target_net(state).max(1)[0].view(1, 1).item(), "- Loss:",
                              loss, "RepMem:", len(agent.replay_memory))

                if agent.steps_done % 600 == 0 and agent.steps_done > 0:
                    with torch.no_grad():
                        print(agent.policy_net(agent.fixed_states[20]))
                        print(agent.policy_net(agent.fixed_states[99]))
                        print("--------------------------------------------------------")

                if done:
                    print("Episode:", i_episode, "- Steps done:", agent.steps_done, "- Episode reward:", episode_reward,
                            "- Episode score:", episode_score, "RepMem:", len(agent.replay_memory))
                    episode_scores.append(episode_score)
                    break

                # Update target policy
                if agent.steps_done % agent.target_update == 0 and agent.steps_done > 0:
                    print("==> Target net updated.")
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())

                # Epoch test
                if agent.steps_done % constants.STEPS_PER_EPOCH == 0 and agent.steps_done > 0:
                    epoch += 1
                    epoch_reward_average, epoch_score_average, n_episodes, \
                    q_values_average = test.test_agent(agent, env.spec.id)
                    information.append(
                        [epoch, agent.steps_done, epoch_reward_average, epoch_score_average, n_episodes,
                         q_values_average])
                    print("INFO:",
                          [epoch, agent.steps_done, epoch_reward_average, epoch_score_average, n_episodes,
                           q_values_average],
                          "Epsilon =", agent.compute_epsilon())

                # Save file periodically
                if agent.steps_done % constants.PERIODIC_SAVE == 0 and agent.steps_done > 0:
                    print("Saving network state...")
                    torch.save(agent.target_net.state_dict(), "info/nn_parameters.pth")
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
        torch.save(agent.target_net.state_dict(), "info/nn_parameters.pth")
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
