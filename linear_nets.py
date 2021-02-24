import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np



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
# two layers
class linear_one(nn.Module):
    def __init__(self):
        super(linear_one, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 500),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x

#three layers
class linear_two(nn.Module):
    def __init__(self):
        super(linear_two, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 500),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Linear(500, 250),
            nn.ReLU(inplace=True),
            nn.Linear(250, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x

# 5 layers
class linear_three(nn.Module):
    def __init__(self):
        super(linear_three, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 800),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Linear(800, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x

# 7 layers
class linear_four(nn.Module):
    def __init__(self):
        super(linear_four, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 800),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Linear(800, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x

# 9 layers
class linear_five(nn.Module):
    def __init__(self):
        super(linear_five, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 900),
            nn.ReLU(inplace=True),
            nn.Linear(900, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 250),
            nn.ReLU(inplace=True),
            nn.Linear(250, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x