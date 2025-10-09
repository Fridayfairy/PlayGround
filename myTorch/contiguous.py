import torch

x = torch.arange(1,25).reshape(2, 3, 4)
print("原始形状:\n", x, x.shape)
y = x.permute(0, 2, 1)
print("permute() ->", y.shape)

try:
    y.view(2, -1)
except:
    print("view() 失败")
    y = y.contiguous()
    print("contiguous() ->", y.shape) #  contiguous() 会返回一个内存连续的张量
    y.view(2, -1)
    print("view() ->", y.shape)