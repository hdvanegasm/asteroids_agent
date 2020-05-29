import random

import gym
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE

import constants
import utils
from agent import device
from agent import get_screen

SHOW_GAME = False
SHOW_GRAPH = True


class DeepQNetwork(torch.nn.Module):

    def __init__(self, height, width, input_channels, outputs):
        super(DeepQNetwork, self).__init__()

        # First layer
        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)

        # Second layer
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Method that computes the number of units of a convolution output given an input
        # Equation taken from:
        # Dumoulin, V., & Visin, F.(2016).A guide to convolution arithmetic for deep learning. 1–31. Retrieved from
        # http://arxiv.org/abs/1603.07285
        def conv2d_output_size(input_size, kernel_size, stride):
            return ((input_size - kernel_size) // stride) + 1

        convw = conv2d_output_size(conv2d_output_size(width, kernel_size=8, stride=4), kernel_size=4, stride=2)
        convh = conv2d_output_size(conv2d_output_size(height, kernel_size=8, stride=4), kernel_size=4, stride=2)

        linear_output_size = 32 * convw * convh

        # Hidden layer
        self.hiden_linear_layer = torch.nn.Linear(linear_output_size, 256)

        # Output layer
        self.head = torch.nn.Linear(256, outputs)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        hidden_out = self.hiden_linear_layer(x)
        x = torch.nn.functional.relu(self.hiden_linear_layer(x))
        return self.head(x), hidden_out


def select_action(state, policy_nn, env):
    epsilon_threshold = constants.TEST_EPSILON
    sample = random.random()
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_nn(state)[0].max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], dtype=torch.long, device=device)


def get_fixed_states():
    fixed_states = []

    env = gym.make('AsteroidsNoFrameskip-v0')
    env.reset()

    cumulative_screenshot = []

    def prepare_cumulative_screenshot(cumul_screenshot):
        # Prepare the cumulative screenshot
        for i in range(constants.N_IMAGES_PER_STATE - 1):
            padding_image = torch.zeros((1, constants.STATE_IMG_HEIGHT, constants.STATE_IMG_WIDTH), device=device)
            cumul_screenshot.append(padding_image)

    prepare_cumulative_screenshot(cumulative_screenshot)

    screen_grayscale_state = get_screen(env)
    cumulative_screenshot.append(screen_grayscale_state.clone().detach())

    N_STATES = 200

    state = utils.process_state(cumulative_screenshot)

    for steps in range(N_STATES + 8):
        if SHOW_GAME:
            env.render()

        action = select_action(state, target_net, env)
        _, _, done, _ = env.step(action.item())  # take a random action

        if done:
            env.reset()
            cumulative_screenshot = []
            prepare_cumulative_screenshot(cumulative_screenshot)

        screen_grayscale = get_screen(env)
        cumulative_screenshot.append(screen_grayscale.clone().detach())
        state = utils.process_state(cumulative_screenshot)
        cumulative_screenshot.pop(0)

        if steps >= 8:
            fixed_states.append(state.clone().detach())

    env.close()
    return fixed_states


def t_sne_algorithm(target_nn):
    states = get_fixed_states()

    N_SAMPLES = 200

    sample_states = random.sample(states, k=N_SAMPLES)

    q_values = []
    flatten_states = []
    for state in sample_states:
        result, hidden = target_nn(state)
        flatten_states.append(hidden[0].clone().detach().tolist())
        q_values.append(result.max(1)[0].view(1, 1).item())
        # print("Q value =", result.max(1)[0].view(1, 1).item())

    flatten_states_tensor = torch.tensor(flatten_states, device=device)

    feat_cols = ['out' + str(i) for i in range(flatten_states_tensor.shape[1])]

    df = pd.DataFrame(flatten_states_tensor.cpu().numpy(), columns=feat_cols)
    df['y'] = q_values

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df[feat_cols].values)

    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(x=df['tsne-2d-one'],
               y=df['tsne-2d-two'],
               c=df['y'],
               s=3,
               edgecolor='',
               cmap="jet")
    if SHOW_GRAPH:
        plt.show()

    plt.savefig("tsne.png")


if __name__ == "__main__":
    env = gym.make('AsteroidsNoFrameskip-v0')

    n_actions = env.action_space.n

    target_net = DeepQNetwork(constants.STATE_IMG_HEIGHT,
                              constants.STATE_IMG_WIDTH,
                              constants.N_IMAGES_PER_STATE // 2,
                              n_actions).to(device)

    target_net.load_state_dict(torch.load("nn_parameters.pth"))
    target_net.eval()

    t_sne_algorithm(target_net)
