# 写一个函数 power_factory(n)，返回一个函数，可以计算某个数的 n 次方。
def power_factory(n):
    def power(x):
        return x**n
    return power

square = power_factory(2)
cube   = power_factory(3)

print(square(4))  # 16
print(cube(2))    # 8
