import sys, os
sys.path.append('c:/Users/moris/OneDrive/kaihatsu_drive/Python/deep_learning')
from dataset.mnist import load_mnist

import pandas as pd

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

df_x_train = pd.DataFrame(x_train)
df_x_train.to_csv('x_train.csv')