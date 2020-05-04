import torch
import torchvision.transforms as transforms
from collections import namedtuple
import numpy as np
#from memory import Transition

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
transitions = [Transition(t1, torch.tensor([[20]]), f1, torch.tensor([2000])),
               Transition(t2, torch.tensor([[30]]), None, torch.tensor([3000])),
               Transition(t3, torch.tensor([[40]]), f3, torch.tensor([4000])),
               Transition(t4, torch.tensor([[50]]), f4, torch.tensor([5000]))]
batch = Transition(*zip(*transitions))

x = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]])
y = torch.tensor([[[[10, 20, 30], [40, 50, 60]]]])
print(torch.cat([x, y], dim=2).shape)
