import torch

@torch.jit.script
def forward(x):
    y = x ** 2 + 3 * x + 1
    return y

print(forward.graph)