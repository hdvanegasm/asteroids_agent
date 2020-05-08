import matplotlib.pyplot
import gym
import torchvision.transforms as transforms
import numpy
import constants
import torch
import utils

import sys


def state_to_image(state, identifier):
    i = 0
    for channel in state[0]:
        matplotlib.pyplot.imsave("state-" + identifier + "-" + str(i) + ".png", channel)
        i += 1


def get_screen(env):
    screen = env.render(mode='rgb_array')
    return transform_image(screen)


def transform_image(screen):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((110, 84)),
        transforms.CenterCrop(84),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.4161, ], [0.1688, ]),
    ])(screen)


def process_state(cumulative_screenshot):
    last_images = cumulative_screenshot[-constants.N_IMAGES_PER_STATE:]

    proccessed_images = []

    for i in range(0, len(last_images), 2):
        first_image = last_images[i]
        second_image = last_images[i + 1]
        join_image = torch.max(first_image, second_image)
        proccessed_images.append(join_image)


    return torch.cat(proccessed_images, dim=0).unsqueeze(0)


env = gym.make('AsteroidsNoFrameskip-v0')
env.reset()

cumulative_screenshots = []

for i in range(50):
    env.render()
    env.step(env.action_space.sample())  # take a random action

    # Con transformacion
    screen = get_screen(env)

    cumulative_screenshots.append(screen)

state = process_state(cumulative_screenshots)
print(state.shape)
state_to_image(state, "Test")

# Sin transformacion
