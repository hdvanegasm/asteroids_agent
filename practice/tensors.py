import torch
import torchvision.transforms as transforms
from collections import namedtuple
import numpy as np
#from memory import Transition

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
t1 = torch.tensor([1, 2, 3])
f1 = torch.tensor([1, 2, 3])

print(t1.max(0))
print(t1)
