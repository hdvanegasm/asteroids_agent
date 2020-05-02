import torch
import torch.nn
import torch.optim
import torch.nn.functional
import torchvision.transforms

import utils

class DeepQNetwork(torch.nn.Module):
    
    def __init__(self, height = 84 * 4, width = 84, outputs = 14):
        super(DeepQNetwork, self).__init__()
        
        # First layer
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.batch_norm1 = torch.nn.BatchNorm2d(16)
        
        # Second layer
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        
        # Convolution output calculation
        convw = utils.conv2d_size_out(utils.conv2d_size_out(utils.conv2d_size_out(width)))
        convh = utils.conv2d_size_out(utils.conv2d_size_out(utils.conv2d_size_out(height)))
        linear_input_size = convw * convh * 32
        
        # Hidden layer
        self.hiden_linear_layer = torch.nn.Linear(linear_input_size, 256)
        
        # Output layer
        self.head = torch.nn.Linear(256, outputs)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.batch_norm1(self.conv1(x)))
        x = torch.nn.functional.relu(self.batch_norm2(self.conv2(x)))
        x = torch.nn.functional.relu(self.hiden_linear_layer(x))
        return self.head(x.view(x.size(0), -1))
    
    