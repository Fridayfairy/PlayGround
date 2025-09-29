from functools import reduce
# 阶乘
print(
    reduce(lambda x, y: x * y, [1,2,3])
)

# 字符串
strs = ["this", "is", "a", "big", "trick"]
print(
    reduce(lambda x, y: x.upper() + "_" + y, strs)
)

# find max
nums = [5, 8, 2, 10, 3]
print(
    reduce(lambda x, y: x if x > y else y, nums)
)

# initializer
print(
    reduce(lambda x, y: x+y, nums, 100)
)