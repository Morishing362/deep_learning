import sys, os
sys.path.append(os.pardir)
from common.functions import *
from comonn.fradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, out_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.randome.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(out_size)

    def 