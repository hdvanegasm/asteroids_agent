import torch

a = torch.tensor([[[1, 2, 3], [4, 5, 6]]]).unsqueeze(0)
print(a.shape)