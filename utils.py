import torch
import torchvision.transforms as transforms

import constants

import matplotlib.pyplot

def transform_image(screen):
    return transforms.Compose([
        transforms.ToPILImage(mode="RGB"),
        transforms.Resize((110, 84)),
        transforms.CenterCrop(84),
        #transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()#,
        #transforms.Normalize([0.4161, ], [0.1688, ]),
    ])(screen)


def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride + 1


def process_state(cumulative_screenshot):
    last_images = cumulative_screenshot[-constants.N_IMAGES_PER_STATE:]

    processed_images = []

    for i in range(0, len(last_images), 2):
        first_image = last_images[i]
        second_image = last_images[i + 1]
        join_image = torch.max(first_image, second_image)
        processed_images.append(join_image)

    return torch.cat(processed_images, dim=0).unsqueeze(0)


def state_to_image(state, identifier):
    i = 0
    for channel in state[0]:
        matplotlib.pyplot.imsave("state-" + identifier + "-" + str(i) + ".png", channel)
        i += 1