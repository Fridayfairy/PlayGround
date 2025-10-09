import torch
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
print(y)
y.backward()
print(x.grad)
print(y.grad)
print(x.grad_fn)
print(y.grad_fn)