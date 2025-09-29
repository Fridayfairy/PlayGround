from cProfile import label
from functools import cached_property
from typing import Sequence, Iterator
from math import sqrt

Number = float

class Vector:
    """ 不可变向量： 索引/切片/迭代/哈希/比较/算数/范数/归一化/夹角/3D外积"""
    def __init__(self, *coords: Number) -> None:
        # import pdb;pdb.set_trace()
        # *coords 会默认识一个参数元组，如果传入的是[1,2,3],则coords ([1,2,3],)
        # 如果传入的是1,2,3,  则coords (1,2,3)
        # support Vector([1,2,3]) Vector(（1,2,3）)形式的序列Sequence
        if len(coords) == 1 and isinstance(coords[0], Sequence) and \
            not isinstance(coords[0], (str, bytes)):
            coords = tuple(coords[0])
        # check
        if any(not isinstance(coord, (int, float)) for coord in coords):
            raise  TypeError("all coords must be int/float")
        # 存成不可变的 tuple，方便哈希与比较。​​实现不可变对象
        object.__setattr__(
            self, "_coords", tuple(float(x) for x in coords)
        )
    
    def __str__(self) -> str:
        return "<" + ", ".join(map(self._fmt, self._coords))+">"

    def __repr__(self) -> str:
        return f"Vector({', '.join(map(self._fmt, self._coords))})"
    
    @staticmethod
    def _fmt(x: float) -> str:
        return f"{x:.6g}" # 紧凑可读,保留6位有效数字

    def __len__(self) -> int:
        return len(self._coords)

    def __iter__(self) -> Iterator[float]:
        return iter(self._coords)

    def __getitem__(self, index):
        # 1. 判断传入的index是不是切片对象（即用户用了切片语法，如 [1:3]）
        # 2. 如果是切片：用切片取self._coords（向量的底层存储，比如元组）的片段
        #    再用*把片段拆成参数，创建一个新的Vector返回（保证返回类型还是Vector）
        # 3. 如果不是切片（即单个索引，如 [2]）：直接返回底层存储的对应元素  
        if isinstance(index, slice):
            return Vector(*self._coords[index])
        return self._coords[index]

    def __eq__(self, other:"Vector") -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self._coords == other._coords

    def __hash__(self) -> int:
        return hash(self._coords)

    def __bool__(self) -> bool:
        return any(v != 0 for v in self._coords)

    def _check_same_dim(self, other: "Vector"):
        if len(self._coords) != len(other._coords):
            raise  ValueError("dim mismatched: {self._coords} vs {len(other._coords)}")

    def __add__(self, other:"Vector") -> "Vector":
        if not isinstance(other, Vector):
            return NotImplemented
        self._check_same_dim(other)
        return Vector(*[a+b for a,b in zip(self, other)])

    def __sub__(self, other:"Vector") -> "Vector":
        if not isinstance(other, Vector):
            return NotImplemented
        self._check_same_dim(other)
        return Vector(*[a-b for a,b in zip(self, other)])

    def __mul__(self, k: float) -> "Vector":
        # 只做标量乘；向量点积使用 __matmul__（@）
        if isinstance(k, (int, float)):
            return Vector(*[a*k for a in self])
        return NotImplemented

    # # 关键：让 数字 * Vector 复用上面的逻辑
    #  “向量乘以数字” 和 “数字乘以向量”
    __rmul__ = __mul__

    def __matmul__(self, other:"Vector") -> "Vector":
        if not isinstance(other, Vector):
            return NotImplemented
        self._check_same_dim(other)
        return Vector(sum(
            a*b for a,b in zip(self, other)
            )
            )

    def __abs__(self) -> float:
        # ||v|| 2
        return sqrt(sum(a*a for a in self))
    
    def __neg__(self) -> "Vector":
        return Vector(-a for a in self)
        
    # 便捷属性 带缓存的属性装饰器
    # 用 @property 把方法变属性
    # @cached_property：解决 “重复计算” 的增强版，第一次调用属性时，执行方法的计算逻辑，把结果缓存起来（存在实例的内存中）
    @cached_property
    def norm(self) -> float:
        return abs(self)

    # 动态坐标别名： x,y,z
    def __getattr__(self, name:str):
        axis = {'x': 0, 'y':1, 'z':2}
        if name in axis:
            idx = axis[name]
            if idx < len(self):
                return self._coords[idx]
        raise AttributeError(name)

   # 不可变：阻止外部设置新属性（保留内部 _coords/cached_property 缓存）
    def __setattr__(self, name, value) -> None:
        raise AttributeError("Vector is immutable") 

    @classmethod
    def zeros(cls, n:int) -> "Vector":
        return cls(*([0]*n))
    
    @staticmethod
    def distance(a:"Vector", b:"Vector") -> float:
        return abs(a-b)

import time

class Timer:
    """with Timer('name') """
    def __init__(self, label:str = "tiktok") -> None:
        self.label = label
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        dt = (time.perf_counter() - self._t0) * 1000
        print(f"[{self.label}] {dt:.2f} ms")
        return False

if __name__ == '__main__': 
    print(Vector.__doc__)

    v = Vector(1,2,3)
    w = Vector(4,5,6)
    print(str(v))
    print(repr(v))

    u = Vector([1,2,3])
    print(str(u))

    # w = Vector([[1,2,3]]) # error
    # print(str(w))

    print(f"len: {len(v)}")
    print("iter:", list(iter(v)))
    print("for iter:", ', '.join([str(a) for a in v]))
    print("getitem:", v[1])
    print("slice:", v[1:3])
    print("u == v: ", u==v)
    print("hash u:", hash(u), "id u:", id(u))
    print("hash v:", hash(v), "id v:", id(v))
    print("bool:", bool(u))
    # print("add:", repr(u+1)) # TypeError: unsupported operand type(s) for +: 'Vector' and 'int'
    print("标量乘，点积", str(u*2.2))
    print("rmul 标量乘，点积", str(2.2 * u))
    print("matmul", u@w)
    print(abs(v), v.norm)
    print("get x y z:", u.x, u.y, u.z)
    # u.x=10  # __setattr__
    print(Vector.distance(u,w))

    with Timer('my timer'):
        for _ in range(1_000_0):
            _ = v@w