import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np

# two layers
class linear_one(nn.Module):
    def __init__(self, dropout):
        super(linear_one, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 500),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(500, 10)
        )
        self.l2 = 0

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x

#three layers
class linear_two(nn.Module):
    def __init__(self, dropout):
        super(linear_two, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 500),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(500, 250),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(250, 10)
        )
        self.l2 = 0

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x

# 5 layers
class linear_three(nn.Module):
    def __init__(self, dropout):
        super(linear_three, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 800),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(800, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(500, 300),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(300, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(200, 10)
        )
        self.l2 = 0

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x

# 7 layers
class linear_four(nn.Module):
    def __init__(self, dropout):
        super(linear_four, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 800),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(800, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(500, 400),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(400, 300),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(300, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(100, 10)
        )
        self.l2 = 0

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x

# 9 layers
class linear_five(nn.Module):
    def __init__(self, dropout):
        super(linear_five, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1000, 900),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(900, 800),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(800, 600),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(600, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(500, 400),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(400, 300),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(300, 250),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(250, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(100, 10)
        )
        self.l2 = 0

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x