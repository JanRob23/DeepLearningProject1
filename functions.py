import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.functional import F
from torch.optim import Adam, SGD
import time
from networks import LeNet5



def train_cnn(model, x, y, x_test, y_test, track_train_test_acc=False, epochs=80, learningRate=0.01, l2_weight_decay=0, batch_size=200):
    start = time.time()
    model = model.float()
    x = torch.from_numpy(x.copy())
    y = torch.from_numpy(y.copy())
    x = x.float()
    y = y.long()
    if torch.cuda.is_available():
        print('yay there is a gpu')
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
    print('I did my training')
    end = time.time()
    print('training took: ', (end-start))
    if track_train_test_acc:
        return model, train_acc, test_acc
    return model, loss_list

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