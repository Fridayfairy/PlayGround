def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a+b

def fib_2():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a+b

if __name__ == '__main__':
    print(list(fib(10)))

    gen = fib_2()
    for _ in range(10):
        print(next(gen), end=' ')