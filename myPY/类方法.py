class PoliceCar():
    ctype: str = "car"
    def __init__(self, user: str, plate: str, use_age: int) -> None:
        self.user = user
        self.plate = plate
        self.use_age =  use_age
    
    @staticmethod
    def check_age(age):
        return age < 5
    
    @classmethod
    def upper_name(cls):
        cls.ctype = cls.ctype.upper() # 只修改类属性
        # cls.user = cls.user.upper() # `user` 是实例属性，不能通过 `cls` 访问
        return cls # 返回类本身（可选），显式声明该方法返回的是类本身，而非 None或其他值，提高代码可读性
        # 后续可以链式调用：PoliceCar.upper_name().some_other_method()


mycar = PoliceCar('ly', 'A19', 4)
print(mycar.ctype)
print(mycar.user)
print(PoliceCar.check_age(7))

# mycar = mycar.upper_name() # 错误， 不要用它修改实例，而是直接调用。这会导致mycar从实例变成了类
print(mycar.ctype)

PoliceCar.upper_name()
print(PoliceCar.ctype)

