import sys, os
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import random

def image_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = \
load_mnist(flatten=True, normalize=False)

r = random.randrange(10000)
img = x_train[r]
label = t_train[r]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

image_show(img)