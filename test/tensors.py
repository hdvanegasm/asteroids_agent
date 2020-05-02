import torch
import torchvision.transforms as transforms
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
t1 = torch.tensor([1, 2, 3])
f1 = torch.tensor([1, 2, 3])

t2 = torch.tensor([1, 2, 3])
f2 = torch.tensor([1, 2, 3])

t3 = torch.tensor([1, 2, 3])
f3 = torch.tensor([1, 2, 3])

t4 = torch.tensor([1, 2, 3])
f4 = torch.tensor([1, 2, 3])
transitions = [Transition(t1, 20, f1, 2000), Transition(t2, 30, None, 3000), Transition(t3, 40, f3, 4000), Transition(t4, 50, f4, 5000)]
batch = Transition(*zip(*transitions))

a = torch.tensor([[1, 2, 3], [1, 2, 3]])
print(a.shape)

print(a.unsqueeze(1))
print(a.unsqueeze(1).shape)