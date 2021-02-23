from networks import LeNet5
import matplotlib.pyplot as plt
from matplotlib.pyplot import pcolor
from fileIO import openMNIST
import numpy as np
import pandas as pd
from functions import train_cnn, eval_cnn, crossvalidationCNN
from networks import LeNet5
from networks import CustomNet
from networks import linear_comb
from plots_and_stuff import plotTrainTestPerformance

def go(train, test):
    print('i am running')
    print('---')
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
    # print('LeNet5')
    # lenet = CustomNet()
    # model, _, _, _ = train_cnn(lenet,x_train, y_train, x_test, y_test, track_train_test_acc=True)
    # acc = eval_cnn(model,x_test, y_test)
    # print(acc)
    print('CustomNet')
    custom = CustomNet()
    train_acc, test_acc, m_list, change = crossvalidationCNN(custom, x_train, y_train, 10)
    plotTrainTestPerformance(train_acc, test_acc, change, m_list)
    # model, _, _ , _= train_cnn(custom, x_train, y_train, x_test, y_test, batch_size=200)
    # acc = eval_cnn(model,x_test, y_test)
    #print(acc)
    # print('linear ensample')
    # linear_co = linear_comb()
    # # # have to adjust this you dont wanna track performance over epochs, just set it to False
    # model, train_acc, test_acc = train_cnn(linear_co, x_train, y_train, x_test, y_test, track_train_test_acc=False)
    # acc = eval_cnn(model, x_test, y_test)
    # print('accuracy on testing:', acc)
    # comment when you dont want plots for epoch
    # plotTrainTestPerformance(train_acc, test_acc, 'Epochs')


if __name__ == "__main__":
    # local path
    train = 'data/mnist_train.csv'
    test = 'data/mnist_test.csv'
    go(train,test)

