import matplotlib.pyplot as plt
from matplotlib.pyplot import pcolor
from fileIO import openMNIST
import numpy as np
from functions import train_cnn


train = '/home/jan/Documents/Deep learning data/mnist_train.csv'
test = '/home/jan/Documents/Deep learning data/mnist_test.csv'

x_train, y_train = openMNIST(train)
x_test, y_test = openMNIST(test)

# reshape and flip data to have it in matrix format
x_train = x_train.reshape(-1, 28, 28, order='C')
x_train = np.flip(x_train[:], 1)
x_test = x_test.reshape(-1, 28, 28, order='C')
x_test = np.flip(x_test[:], 1)
# confirmation plots
# print(y_test[600])
# fig = pcolor(x_test[600], cmap='gist_gray')
# plt.show()
train_cnn(x_train, y_train, 'lenet')
