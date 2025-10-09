import queue
import threading
import time

def producer(q):
    for i in range(5):
        item = f"Item {i}"
        q.put(item) # 投递任务 队列内部计数器 unfinished_tasks += 1
        print(f"Produced {item}")
        time.sleep(0.1)
    q.put(None) # 通知哨兵结束

def consumer(q):
    while True:
        item = q.get() # 取出任务
        if item is None:
            q.task_done() # 对哨兵值也要调用 task_done()
            break
        print(f"Consumed: {item}")
        time.sleep(0.2)
        q.task_done() # 取出的该任务完成， 队列内部计数器unfinished_tasks -= 1
q = queue.Queue(maxsize=5)

prod_thread = threading.Thread(target=producer, args=(q,))
cons_thread = threading.Thread(target=consumer, args=(q,))

prod_thread.start()
cons_thread.start()

prod_thread.join()
# cons_thread.join() # 不需要，q.join() 已经提供了足够的同步保证
q.join() # 阻塞主线程，直到队列里所有任务都标记为完成（即 unfinished_tasks == 0）
print("Done!")

