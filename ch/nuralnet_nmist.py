import sys, os
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
import pandas as pd

import sigmoid_func
import softmax_func

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test
    # normalizeは正規化、falttenは一次元配列化、one_hot_labelはラベルを特徴量(0 or 1)にするか数値(0, 1, 2,...)にするか。

# # csv書き出し
# def data_to_csv(x_test, t_test):
#     df = pd.DataFrame(x_test, t_test)
#     df.to_csv('x-t_test.csv')

def init_network():
    with open(os.path.join(os.getcwd(),'ch/sample_weight.pkl'), 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):  
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_func.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_func.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_func.softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range (len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print('Accuracy : ', float(accuracy_cnt)/len(x))

# x_test, t_test = get_data()
# data_to_csv(x_test, t_test)