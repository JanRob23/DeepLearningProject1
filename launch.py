from typing import Coroutine
from networks import LeNet5
import matplotlib.pyplot as plt
from matplotlib.pyplot import pcolor
from fileIO import openMNIST
import numpy as np
import pandas as pd
from functions import train_cnn, eval_cnn, crossvalidationCNN, train_linear_models_plus_average, test_model
from networks import LeNet5
from networks import CustomNet
from networks import linear_comb
from plots_and_stuff import plotTrainTestPerformance

def go(train, test):
    print('i am running')
    print('---')
    x_train, y_train, x_test, y_test = reshape_data(train, test)
    
    #-----------LeNet 5 ---------------#
    # print('LeNet5')
    # leNet = LeNet5()

    # epoch_eval_single(lenet, x_train, y_train, x_test, y_test)
    # cross_val(lenet, x_train, y_train, x_test, y_test)
    #test_model(leNet, x_train, y_train, x_test, y_test)


    #----------- CustomNet --------------#
    # print('CustomNet')
    # custom = CustomNet()

    # epoch_eval_single(custom, x_train, y_train, x_test, y_test)
    # cross_val(custom, x_train, y_train, x_test, y_test)
    # test_model(custom, x_train, y_train, x_test, y_test)


    #----------- Linear --------------#
    # print('linear ensample')
    #linear_co = linear_comb()
    # # have to adjust this you dont wanna track performance over epochs, just set it to False
    #model, train_acc, test_acc = train_cnn(linear_co, x_train, y_train, x_test, y_test, track_train_test_acc=False)
    # acc = eval_cnn(model, x_test, y_test)
    # print('accuracy on testing:', acc)
    # comment when you dont want plots for epoch
    # plotTrainTestPerformance(train_acc, test_acc, 'Epochs')
    print("Linear Nets")
    val = 0
    for i in range(0,10):
        print("L2 reg value: ", val)
        train_linear_models_plus_average(x_train, y_train, x_test, y_test, track_train_test_acc=False, l2=val)
        val += 0.0025

def reshape_data(train, test):
    x_train, y_train = openMNIST(train)
    x_test, y_test = openMNIST(test)

    # reshape and flip data to have it in matrix format
    x_train = x_train.reshape(-1, 28, 28, order='C')
    x_train = np.flip(x_train[:], 1)
    x_test = x_test.reshape(-1, 28, 28, order='C')
    x_test = np.flip(x_test[:], 1)
    return x_train, y_train, x_test, y_test

def confirmation_plots(x_test, y_test):
    print(y_test[600])
    fig = pcolor(x_test[600], cmap='gist_gray')
    plt.show()

def epoch_eval_single(model, x_train, y_train, x_test, y_test):
    model, train_acc, test_acc = train_cnn(model ,x_train, y_train, x_test, y_test, track_train_test_acc=True)
    acc = eval_cnn(model,x_test, y_test)
    print(acc)
    plotTrainTestPerformance(train_acc, test_acc, 'Epochs')

def cross_val(model, x_train, y_train, x_test, y_test):
    train_acc, test_acc, m_list, change = crossvalidationCNN(model, x_train, y_train, 5)
    plotTrainTestPerformance(train_acc, test_acc, change, m_list)

if __name__ == "__main__":
    # local path
    train = 'data/mnist_train.csv'
    test = 'data/mnist_test.csv'
    go(train,test)

