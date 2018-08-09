import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    h = 1e-5
    return (f(x+h) - f(x)) / h

def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5.000000))


