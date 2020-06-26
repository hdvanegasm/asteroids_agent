import torch
import torchvision.transforms as transforms
from collections import namedtuple
import numpy as np

# from memory import Transition

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.ones(size=(1, 84, 84), device=device)
print(torch.cuda.memory_allocated(device))
print(device)