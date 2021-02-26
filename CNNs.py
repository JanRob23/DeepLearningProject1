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
