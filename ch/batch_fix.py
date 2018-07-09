import sys, os
import nuralnet_nmist as nn
import numpy as np

x, t = nn.get_data()
network = nn.init_network()

batch_size = 100
accuracy = 0

# 並列化
for i in range (0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = nn.predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy += np.sum(p == t[i:i+batch_size])

print('Accuracy : ', float(accuracy)/len(x))