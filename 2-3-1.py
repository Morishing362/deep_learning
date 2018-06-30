#2-3-1
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    temp = x1*w1 + x2*w2
    if temp <= theta:
        return 0
    if temp > theta:
        return 1

a = input().split()
a[0], a[1] = float(a[0]), float(a[1])
b = AND(a[0], a[1])
print(b)
