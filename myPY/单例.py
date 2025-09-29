from threading import Lock

class Singleton():
    _instance = None
    _lock = Lock() # 为了线程安全
    _inited = False # 不加的话，每次调用都会init，
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None: # 双重检查锁定（Double-Checked Locking）
                    cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self, x) -> None:
        if not self._inited:
            self._inited = True # 仅初始化一次
            ## 正常的init代码 ↓
            self.x = x
    
    def __eq__(self, other):
        if isinstance(other, Singleton):
            return self.x == other.x
        return False

model_a = Singleton(3)
model_b = Singleton(4)
print(model_a is model_b) 
print(model_a == model_b) 
print(model_a.x, model_b.x) 
