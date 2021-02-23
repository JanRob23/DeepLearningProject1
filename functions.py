import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.functional import F
from torch.optim import Adam, SGD
import time
from networks import LeNet5
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold



def train_cnn(model, x, y, x_test, y_test, track_train_test_acc=False, epochs=50, learningRate=0.01, l2_weight_decay=0.001, batch_size=200):
    start = time.time()
    model = model.float()
    x = torch.from_numpy(x.copy())
    y = torch.from_numpy(y.copy())
    x = x.float()
    y = y.long()
    if torch.cuda.is_available():
        #print('yay there is a gpu')
        model = model.cuda()
        x = x.cuda()
        y = y.cuda()
    optimizer = Adam(model.parameters(), lr=learningRate, weight_decay=l2_weight_decay)
    criterion = nn.CrossEntropyLoss()
    loss_list = []
    # batch sizes are claculated and sorted, by defaut batchsize is entire dataset
    if not batch_size or batch_size > x.shape[0]:
        batch_size = x.shape[0]
    batch_num = x.shape[0] / batch_size
    x = x.reshape(-1, batch_size, 1, 28, 28)
    y = y.reshape(-1, batch_size)
    test_acc = []
    train_acc = []
    for epoch in range(0, epochs):
        # loop over the number of batches feeds in batch_size many images and performs backprob
        # then again and so on
        for i in range(0, int(batch_num)):
            # Here we feed in training data and perform backprop according to the loss
            # Run the forward pass
            outputs = model.forward(x[i])
            loss = criterion(outputs, y[i])
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # here the training and testing acc is tracked if indicated to the function
        if track_train_test_acc:
            train_acc.append(eval_cnn(model, x, y))
            test_acc.append(eval_cnn(model, x_test, y_test))
    #print('I did my training')
    end = time.time()
    #print('training took: ', (end-start))
    return model, train_acc, test_acc

def eval_cnn(model, x, y):
    x = x.reshape(-1, 1, 28, 28)
    # this makes sure that eval_cnn can be called with both training and testing data and the diff types, shapes
    if not torch.is_tensor(x):
        x = torch.from_numpy(x.copy())
        y = torch.from_numpy(y.copy())
    else:
        y = y.reshape(-1)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    x = x.float()
    y = y.long()
    output = model.forward(x)
    total = y.size(0)
    _, predicted = torch.max(output.data, 1)
    correct = 0

    for i in range(total):
        if predicted[i] == y[i]:
            correct += 1
    return correct / total


def crossvalidationCNN(model_used, x, y, k):
    # setup the k-fold split
    total = x.shape[0]
    bin_size = int(total / k)
    folds_x = np.array(np.array_split(x, k))
    folds_y = np.array(np.split(y, k))
    acc_train_m = list()
    acc_test_m = list()
    m_list = list()

    # define m range in this case m corresponds with epochs
    # to change what is going to vary with m, mention in the train_cnn function
    # eg. learning_rate = m, batch_size = m ...
    # also declare what you change for the graph legend
    # type 'architecture' if changing architecture, make there only be 1 step 
    change = 'l2 regularization'
    start = 0.005
    stop = 0.008
    step = 0.0001

    # new folder for each new run, except if ran size is 1
    # file with list of ave accuracies
    # plot 
    best_m = 0
    best_m_train = 0
    best_m_acc = 0
    m_range = np.arange(start, stop, step)
    print(f'training and evaluating {k * len(m_range)} models')

    for m in tqdm(m_range, desc='m values', position=0):  # loop over given m settings
        # mFile = f'{newfile}/{change}_{m}'
        # os.makedirs(mFile, exist_ok= True)
        acc_train = list()
        acc_test = list()
        kf = KFold(n_splits=k)
        for train, test in kf.split(x):
            train_x, test_x, train_y, test_y = x[train], x[test], y[train], y[test]  # train a new model for each fold and for each m
            model , train_acc, test_acc = train_cnn(model_used, train_x, train_y, test_x, test_y, l2_weight_decay=m, batch_size = 64, epochs=50)
            acc = eval_cnn(model, train_x, train_y)
            acc_train.append(acc)
            acc = eval_cnn(model, test_x, test_y)
            acc_test.append(acc)
        mean_train_acc = round(np.mean(acc_train), 4)
        mean_test_acc = round(np.mean(acc_test), 4)
        acc_train_m.append(mean_train_acc)
        acc_test_m.append(mean_test_acc)
        if mean_test_acc > best_m_acc:
            best_m_acc = mean_test_acc
            best_m_train = mean_train_acc
            best_m = m
        m_list.append(round(m, 4))
    print(f'\nBest m: {best_m}\ntrain acc: {best_m_train}\ntest acc:{best_m_acc}')
    return acc_train_m, acc_test_m, m_list, change

