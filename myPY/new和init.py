class Line:
    def __new__(cls, *args, **kwargs):
        print(f"new: {args} {kwargs}")
        instance = super().__new__(cls) # 真正创建对象
        return instance
    
    def __init__(self, length) -> None:
        print("init")
        self.length = length
    
line = Line(10)