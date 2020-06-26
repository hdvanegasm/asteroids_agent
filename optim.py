import torch 
import torch.optim


class RMSprop(torch.optim.Optimizer):

	def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, momentum=0):
		defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps)
		super(RMSprop, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(RMSprop, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('momentum', 0)

	def step(self, closure=None):
		with torch.no_grad():
			for group in self.param_groups:
				for param in group["params"]:
					state = self.state[param]

					if len(state) == 0:
						state["squared_grad"] = torch.zeros_like(param, memory_format=torch.preserve_format)
						state["avg_grad"] = torch.zeros_like(param, memory_format=torch.preserve_format)
						state["momentum"] = torch.zeros_like(param, memory_format=torch.preserve_format) 

					grad_param = param.grad

					squared_grad = state["squared_grad"]
					avg_grad = state["avg_grad"]
					momentum_buff = state["momentum"]

					squared_grad = group['alpha'] * squared_grad + (1 - group['alpha']) * torch.mul(grad_param, grad_param)
					avg_grad = group['alpha'] * avg_grad + (1 - group['alpha']) * grad_param

					momentum_buff = group['momentum'] * momentum_buff - \
						group['lr'] * grad_param / (torch.sqrt(squared_grad - torch.mul(avg_grad, avg_grad) + group['eps']))

					new_param = param + momentum_buff
					param.copy_(new_param)
