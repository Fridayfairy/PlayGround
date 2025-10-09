import torch

x = torch.arange(1,13).reshape(3, 4)
print("原始形状:\n", x, x.shape)

# view()
y1 = x.view(4, 3)
print("\nview() ->", y1.shape)

# reshape()
y2 = x.reshape(2, 6)
print("reshape() ->", y2.shape)

# permute()
y3 = x.permute(1, 0)
print("permute() ->", y3.shape)

# transpose()
y4 = x.transpose(0, 1)
print("transpose() ->", y4.shape)
