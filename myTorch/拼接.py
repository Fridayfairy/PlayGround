# cat
# stack

import torch

# 创建两个 Tensor
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# torch.cat: 沿现有维度拼接（dim=0 或 dim=1）
cat0 = torch.cat((a, b), dim=0)
cat1 = torch.cat((a, b), dim=1)

print("原始 a 形状:", a.shape)
print("cat(dim=0):\n", cat0, cat0.shape)
print("cat(dim=1):\n", cat1, cat1.shape)

# torch.stack: 在新维度上堆叠
stack0 = torch.stack((a, b), dim=0)
stack1 = torch.stack((a, b), dim=1)

print("\nstack(dim=0):\n", stack0, stack0.shape)
print("stack(dim=1):\n", stack1, stack1.shape)
