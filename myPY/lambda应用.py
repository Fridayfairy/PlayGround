# 匿名小函数
f = lambda x, y : x + y
print(f(3, 4))

# 排序
nums = [(1, 'b'), (2, 'a'), (3, 'c')]
print(sorted(nums, key=lambda x: x[1]))
print(sorted(nums, key=lambda x: x[0]))

# 组合 map filter reduce
nums = [1, 2, 3, 4]
print(list(
    map(lambda x: x**2, nums)
))

print(list(
    filter(lambda x: x > 2, nums)
))

from functools import reduce
print(
    reduce(lambda x, y: x + y, nums)
)

