from shutil import RegistryError
import time
from functools import wraps

def retry(times, delay):
    def deco(func):
        @wraps(func) # 保留原函数的元信息
        def doSth(*args, **kwargs):
            exc = None
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    exc = e
                    print(f"[retry] {func.__name__} failed, retry {i+1}")
                    time.sleep(delay)
            raise exc
        return doSth
    return deco


@retry(times=3, delay=1)
def work():
    import random
    if random.random() < 0.7:
        raise ValueError("fail")
    return "success"

print(work())