import numpy as np
# import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# a = np.linspace(-5.0, 5.0, 100)
# b = sigmoid(a)

# plt.plot(a, b)
# plt.show()