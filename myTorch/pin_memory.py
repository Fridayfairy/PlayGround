import torch, time
x = torch.randn(10000, 1000)

# 普通内存
t1 = time.time()
for _ in range(1000):
    _ = x.to('cuda')
print("Normal:", time.time() - t1)

# 锁页内存
x = x.pin_memory()
t2 = time.time()
for _ in range(1000):
    _ = x.to('cuda', non_blocking=True)
print("Pinned:", time.time() - t2)