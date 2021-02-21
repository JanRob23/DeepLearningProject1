import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.functional import F
from torch.optim import Adam, SGD

from cnn_functions import LeNet5



def train_cnn(x, y, net, epochs=20, learningRate=0.007, l2_weight_decay=0.001, batch_size=20):
    #if net == 'LeNet5':
    model = LeNet5()
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if torch.cuda.is_available():
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
    x = x.reshape(-1, batch_size, 1, 16, 15)
    y = y.reshape(-1, batch_size)

    for epoch in range(0, epochs):
        # loop over the number of batches feeds in batch_size many images and performs backprob
        # then again and so on
        for i in range(0, int(batch_num)):
            # Here we feed in training data and perform backprop according to the loss
            # Run the forward pass
            outputs = model.forward(x[i])
            y = y.long()
            loss = criterion(outputs, y[i])
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, loss_list