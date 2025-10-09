from queue import PriorityQueue

q = PriorityQueue()
q.put((2, 'no.2'))
q.put((1, 'no.1'))
print(q.get())