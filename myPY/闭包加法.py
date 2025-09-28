def add_x(x):
    def add_y(y):
        return x + y
    return add_y
add_10 = add_x(10)
print(add_10(5))
print(add_10(10))