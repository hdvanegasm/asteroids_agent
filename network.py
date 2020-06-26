import torch
import torch.nn
import torch.nn.functional
import torch.optim


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
        x = torch.nn.functional.relu(self.hiden_linear_layer(x))
        return self.head(x)
