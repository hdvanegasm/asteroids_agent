import torch
import torchvision.transforms as transforms
from collections import namedtuple
import numpy as np

# from memory import Transition

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.zeros((2, 3), device=device)
a2 = a.clone()
print(a)
print(a2)