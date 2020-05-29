import torch
import torchvision.transforms as transforms
from collections import namedtuple
import numpy as np

# from memory import Transition

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device).total_memory)
print(device)