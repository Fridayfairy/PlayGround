import torch
import numpy as np

# NumPy → Tensor
a = np.array([[1, 2], [3, 4]], dtype=np.float32)
t1 = torch.from_numpy(a)
t2 = torch.tensor(a)
t3 = torch.as_tensor(a)

print("原始NumPy数组:\n", a)
print("from_numpy共享内存:\n", t1)
print("tensor拷贝数据:\n", t2)
print("as_tensor自动判断:\n", t3)

# 修改NumPy数组，看Tensor是否受影响
a[0, 0] = 100
print("\n修改NumPy后:")
print("a:", a)
print("t1(from_numpy):", t1)   # 会变
print("t2(torch.tensor):", t2) # 不会变

# Tensor → NumPy
t = torch.tensor([[5, 6], [7, 8]])
np_t = t.numpy()
print("\nTensor 转 NumPy:", np_t)

# GPU Tensor
t_gpu = torch.tensor([1.0, 2.0], device='cuda')
np_gpu = t_gpu.detach().cpu().numpy()
print("GPU Tensor 转 NumPy:", np_gpu)