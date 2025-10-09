import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

均匀分布 = torch.rand(2, 3) # 【0, 1）之间的均匀分布
print(均匀分布)

标准正态分布 = torch.randn(2, 3) # 标准正态分布（均值为0，标准差为1）
print(标准正态分布)

整数均匀分布 = torch.randint(0, 10, (2, 3)) # 【0, 10）之间的整数均匀分布
print(整数均匀分布)

随机排列 = torch.randperm(5) # 随机排列（0到4）
print(随机排列)

x =torch.zeros(2,3)
print("rand like:", torch.rand_like(x))
print("randn like:", torch.randn_like(x))
