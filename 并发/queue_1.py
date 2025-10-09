from queue import Queue
q = Queue(maxsize=5)
q.put(1)
q.put(2.0)
q.put('3')
q.put(None) # 通知哨兵结束， get 到None时，可以退出。自身占了 maxsize中的一个位置

c=q.get()
print(c, type(c))

c=q.get()
print(c, type(c))

c=q.get()
print(c, type(c))

c=q.get() # 此处取不到，会阻塞等待ing， forever
print(c, type(c))