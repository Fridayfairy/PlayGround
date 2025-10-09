# 双向通信原理​
def echo():
    while True:
        x = yield # yield 等待外部 send
        print("Say: ", x)

gen = echo() # 创建生成器对象 gen，此时函数内的代码​​尚未执行​​。
next(gen) # 启动生成器​​，执行 echo()函数直到遇到第一个 yield； 执行到 x = yield时，生成器​​暂停​​，并将控制权交还给调用者；此时 x还未被赋值（因为 yield右边没有值）
gen.send('hi') # 将 "hello"发送给生成器，​​从上次暂停的 yield处恢复执行​​。"hello"会赋值给 x，然后执行 print，循环再次执行到 x = yield，生成器再次暂停。
gen.send(2025)

def demo():
    try:
        yield 1 # 第一次暂停点.
    except ValueError: # 捕获外部 throw 的异常
        print("find error")
    finally:
        print("end")
    yield 2 # 第二次暂停点
    yield 3

g = demo()
print(next(g)) # 输出: 1 （执行到 yield 1）

print(g.throw(ValueError)) # 输出: 内部捕获 ValueError → 2 
# ​​向生成器内部抛出 ValueError异常​​，生成器从上次暂停的 yield 1处恢复执行。
# 由于 yield 1在 try块中，异常被捕获，执行 except ValueError块中的代码（打印消息）。
# 继续执行到 yield 2，生成器再次暂停，返回 2。

print(next(g)) 
g.close()