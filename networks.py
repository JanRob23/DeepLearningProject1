import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.layers_conv = self.LeNetLayersConv()
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
        x = x.view(-1, 16 * 8 * 8)
        x = self.layers_linear(x)
        return x

    def CustomNetLayersConv(self):
        layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=4, out_channels=6, kernel_size=4, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=4, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
            )
        return layers

    def CustomNetLayersLinear(self):
        layers = nn.Sequential(
            nn.Linear(16 * 8 * 8, 120),
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
            nn.Linear(28*28, 250),
            nn.Tanh(),
            nn.Linear(250, 10)
        )
        return layers
    def linear_three(self):
        layers = nn.Sequential(
            nn.Linear(28 * 28, 400),
            #nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(400, 150),
            nn.Tanh(),
            #nn.ReLU(inplace=True),
            nn.Linear(150, 10)
        )
        return layers
    def linear_four(self):
        layers = nn.Sequential(
            nn.Linear(28 * 28, 500),
            #nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(500, 300),
            #nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(300, 150),
            #nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(150, 10)
        )
        return layers
    def linear_six(self):
        layers = nn.Sequential(
            nn.Linear(28 * 28, 600),
            #nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(600, 350),
            #nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(350, 200),
            #nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(200, 100),
            #nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(100, 60),
            nn.Tanh(),
            nn.Linear(60, 10)
        )
        return layers
    def linear_eight(self):
        layers = nn.Sequential(
            nn.Linear(28 * 28, 600),
          #  nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(600, 400),
           # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(400, 300),
           # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(300, 400),
           # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(400, 150),
            #nn.ReLU(inplace=True),
            nn.Linear(150, 100),
            #nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(100, 50),
           # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(50, 10)
        )
        return layers