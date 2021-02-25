import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np



class LeNet5(torch.nn.Module):

    def __init__(self, drop = False):
        super(LeNet5, self).__init__()
        self.layers_conv = self.LeNetLayersConv()
        self.drop = drop
        if self.drop:
            self.layers_linear = self.LeNetLayersLinearDrop()
        else:
            self.layers_linear = self.LeNetLayersLinear()
        

    def forward(self, x):
        x = self.layers_conv(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.layers_linear(x)
        return x

    def LeNetLayersConv(self):
        layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
            )
        return layers

    def LeNetLayersLinearDrop(self):
        layers = nn.Sequential(
            nn.Dropout(p = 0.25),
            nn.Linear(16 * 5 * 5, 120),
            nn.Dropout(p = 0.25),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )
        return layers
    def LeNetLayersLinear(self):
        layers = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )
        return layers

class CustomNet(torch.nn.Module):

    def __init__(self):
        super(CustomNet, self).__init__()
        self.layers_conv = self.CustomNetLayersConv()
        self.layers_linear = self.CustomNetLayersLinear()

    def forward(self, x):
        x = self.layers_conv(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.layers_linear(x)
        return x

    def CustomNetLayersConv(self):
        layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=4, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
            )
        return layers

    def CustomNetLayersLinear(self):
        layers = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )
        return layers


class linear_comb(nn.Module):
    def __init__(self):
        super(linear_comb, self).__init__()
        self.linear_two1 = self.linear_two()
        self.linear_three1 = self.linear_three()
        self.linear_four1 = self.linear_four()
        self.linear_six1 = self.linear_six()
        self.linear_eight1 = self.linear_eight()
        self.readout_layer = nn.Linear(5*10, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        out1 = self.linear_two1(x)
        out2 = self.linear_three1(x)
        out3 = self.linear_four1(x)
        out4 = self.linear_six1(x)
        out5 = self.linear_eight1(x)
        out_all = torch.cat((out1, out2, out3, out4, out5), 1)
       # print(out_all.shape)
        x = self.readout_layer(out_all)
        return x
    def linear_two(self):
        layers = nn.Sequential(
            nn.Linear(28*28, 500),
            #nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10)
        )
        return layers
    def linear_three(self):
        layers = nn.Sequential(
            nn.Linear(28 * 28, 800),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(800, 300),
            #nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Linear(300, 10)
        )
        return layers

    def linear_four(self):
        layers = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(500, 300),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(300, 10)
        )
        return layers
    def linear_six(self):
        layers = nn.Sequential(
            nn.Linear(28 * 28, 1200),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(1200, 700),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(700, 400),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(200, 100),
            #nn.Tanh(),
            nn.Linear(100, 10)
        )
        return layers
    def linear_eight(self):
        layers = nn.Sequential(
            nn.Linear(28 * 28, 1200),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(1200, 800),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(800, 600),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(600, 500),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(500, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 200),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Linear(100, 10)
        )
        return layers