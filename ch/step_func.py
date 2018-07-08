import numpy as np
# import matplotlib.pyplot as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)

# a = np.linspace(-5.0, 5.0, 100)
# b = step_function(a)

# plt.plot(a, b)
# plt.show()