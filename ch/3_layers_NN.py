import numpy as np
import sigmoid_func

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.3,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

def forward(netwrok, x):
    W1, W2, W3 = netwrok['W1'], netwrok['W2'], netwrok['W3']
    b1, b2, b3 = netwrok['b1'], netwrok['b2'], netwrok['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_func.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_func.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = sigmoid_func.sigmoid(a3)

    return y

network = init_network()
x = np.array([0.1, 0.5])
print(forward(network, x))