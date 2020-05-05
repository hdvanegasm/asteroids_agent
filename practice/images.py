import matplotlib.pyplot
import gym
import torchvision.transforms as transforms
from PIL import Image
import numpy

def get_screen(env):
    screen = env.render(mode='rgb_array')
    return transform_image(screen)

def transform_image(screen):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((110, 84)),
        transforms.CenterCrop(84),
        transforms.Grayscale(num_output_channels=1)#,
        #transforms.ToTensor(),
        #transforms.Normalize([0.4161, ], [0.1688, ]),
    ])(screen)

env = gym.make('Asteroids-v0')
env.reset()
for _ in range(52):
    env.render()
    env.step(env.action_space.sample()) # take a random action

screen = get_screen(env)

#screen.save("prueba.png")
array_screen = numpy.asarray()

