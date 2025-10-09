import torch
x=torch.tensor([2.0],requires_grad=True)
y = x**3
grad_x, = torch.autograd.grad(y, x, create_graph=True)
print(grad_x)

grad2_x, = torch.autograd.grad(grad_x, x)
print(grad2_x)
