import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class LeNet(nn.Module):
    def __init__(self, n_out):
        super(LeNet, self).__init__()
    
        layers = []
        layers.append(nn.Conv2d(1, 6, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(nn.Conv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        
        layers.append(nn.Conv2d(16, 120, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        
        self.main = nn.Sequential(*layers)
        
        layers = []
        layers.append(nn.Linear(120, 84))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(84, n_out))
        
        self.fc_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120)
        return self.fc_layers(x).squeeze()


# class LeNet(MetaModule):
#     def __init__(self, n_out):
#         super(LeNet, self).__init__()
    
#         layers = []
#         layers.append(nn.Conv2d(1, 6, kernel_size=5))
#         layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

#         layers.append(nn.Conv2d(6, 16, kernel_size=5))
#         layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        
#         layers.append(nn.Conv2d(16, 120, kernel_size=5))
#         layers.append(nn.ReLU(inplace=True))
        
#         self.main = nn.Sequential(*layers)
        
#         layers = []
#         layers.append(nn.Linear(120, 84))
#         layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Linear(84, n_out))
        
#         self.fc_layers = nn.Sequential(*layers)
        
#     def forward(self, x):
#         x = self.main(x)
#         x = x.view(-1, 120)
#         return self.fc_layers(x).squeeze()