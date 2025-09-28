# funcs = []
# for i in range(3):
#     funcs.append(lambda: i)

funcs = [lambda i=i: i for i in range(3)]

for f in funcs:
    print(f())
