from test_folder import print_stuff
from fileIO import openMNIST


train = 'data/mnist_train.csv'
test = 'data/mnist_test.csv'
x_train, y_train = openMNIST(train)
x_test, y_test = openMNIST(test)

