import matplotlib.pyplot
import gym
import torchvision.transforms as transforms
import numpy
import constants
import torch

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
    last_four_images = cumulative_screenshot[-constants.N_IMAGES_PER_STATE:]
    return torch.cat(last_four_images, dim=0).unsqueeze(0)


env = gym.make('AsteroidsNoFrameskip-v0')
env.reset()

cumulative_screenshots = []

for i in range(50):
    env.render()
    env.step(env.action_space.sample()) # take a random action

    # Con transformacion
    screen = get_screen(env)
    if i > 20:
        matplotlib.pyplot.imsave("image_transform" + str(i) + ".png", screen[0])

    cumulative_screenshots.append(screen)

state = process_state(cumulative_screenshots)
print(state.shape)
state = process_state(cumulative_screenshots)
matplotlib.pyplot.imsave("state.png", state[0][0])



# Sin transformacion

