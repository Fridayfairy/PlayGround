import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            cost = (time.perf_counter() - start_time) * 1000
            print(f"[timer] {func.__name__} took time {cost:.2f} ms")
    return wrapper

@timer
def add(x, y):
    time.sleep(0.05)
    return x+y

print(add(3, 4))
