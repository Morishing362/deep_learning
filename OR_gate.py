import numpy as np

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    temp = np.sum(x*w) + b
    if temp <= 0:
        return 0
    if temp > 0:
        return 1

a = input().split()
a[0], a[1] = float(a[0]), float(a[1])
print(OR(a[0], a[1]))