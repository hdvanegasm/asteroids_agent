import torch
import torchvision.transforms as transforms
from collections import namedtuple
import numpy as np

# from memory import Transition

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

a = torch.tensor([[100, 200, 300], [400, 500, 600], [400, 500, 600]])
actions = np.concatenate(torch.tensor([[2], [1], [0]]).numpy())

indices = np.array([0, 2, 1])

print(a[indices, actions])