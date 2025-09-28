from functools import wraps

def log(func):
    @wraps(func)
    def _log(*args, **kwargs):
        print(f"[info] {func.__name__} args: {args}, kwargs: {kwargs}")
        res = func(*args, **kwargs)
        print(f"[info] res: {res}")
        return res
    return _log

@log
def func1(x, y):
    return x+y

a=func1(3, 4)
print(a)