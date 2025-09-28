def make_accumulator():
    sum = 0
    def acc(x):
        nonlocal sum
        sum += x
        return sum
    return acc

acc = make_accumulator()
print(acc(10))  # 10
print(acc(5))   # 15
print(acc(7))   # 22
