import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y ** 3

# y.backward()       # 第一次 backward
y.backward(retain_graph=True)       # 保留图 则可以2次 backward
try:
    z.backward()   # 第二次 backward（报错）
    # z.backward()   # 第3次 backward（报错）

except Exception as e:
    print("错误:", e)