import sys, os
import nuralnet_nmist as nn

x, t = nn.get_data()
network = nn.init_network()

W1, W2, W3 = network['W1'], network['W2'], network['W3']

print(x.shape)
print(x[0].shape)
print(W1.shape,'\n', W2.shape,'\n', W3.shape)
