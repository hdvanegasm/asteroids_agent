import math
import random

import gym

import torch.optim
import torch.nn.functional

from network import DeepQNetwork
import constants
import memory
import utils


def select_action(state, policy_nn, steps_done, env):
    epsilon_threshold = constants.EPS_END + \
                        (constants.EPS_START - constants.EPS_END) * \
                        math.exp(-1 * steps_done / constants.EPS_DECAY)
    sample = random.random()
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_nn(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], dtype=torch.long)


def optimize_model(target_nn, policy_nn, memory, optimizer):
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
    loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    for param in policy_nn.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()


def get_screen(env):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    return utils.transform_image(screen)


def main_training_loop():
    env = gym.make('Asteroids-v0')

    n_actions = env.action_space.n

    policy_net = DeepQNetwork(constants.STATE_IMG_HEIGHT * constants.N_IMAGES_PER_STATE,
                              constants.STATE_IMG_WIDTH,
                              n_actions)

    target_net = DeepQNetwork(constants.STATE_IMG_HEIGHT * constants.N_IMAGES_PER_STATE,
                              constants.STATE_IMG_WIDTH,
                              n_actions)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.RMSprop(policy_net.parameters())
    replay_memory = memory.ReplayMemory(constants.REPLAY_MEMORY_SIZE)

    steps_done = 0

    for i_episode in range(constants.N_EPISODES):
        cumulative_screenshot = []

        # Prepare the cumulative screenshot
        padding_image = torch.zeros((1, constants.STATE_IMG_HEIGHT, constants.STATE_IMG_WIDTH))
        for i in range(constants.N_IMAGES_PER_STATE - 1):
            cumulative_screenshot.append(padding_image)

        env.reset()
        episode_reward = 0

        screen_grayscale_state = get_screen(env)
        cumulative_screenshot.append(screen_grayscale_state)

        state = utils.process_state(cumulative_screenshot)

        for i in range(constants.N_TIMESTEP_PER_EP):
            env.render()
            action = select_action(state, policy_net, steps_done, env)
            _, reward, done, _ = env.step(action)
            episode_reward += reward

            reward_tensor = torch.tensor([reward])

            screen_grayscale = get_screen(env)
            cumulative_screenshot.append(screen_grayscale)

            if done:
                next_state = None
            else:
                next_state = utils.process_state(cumulative_screenshot)

            replay_memory.push(state, action, next_state, reward_tensor)

            state = next_state

            optimize_model(target_net, policy_net, replay_memory, optimizer)
            steps_done += 1

            if done:
                print("Episode reward:", episode_reward, "Steps done:", steps_done)
                break

        if i_episode % constants.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()

if __name__ == "__main__":
    main_training_loop()