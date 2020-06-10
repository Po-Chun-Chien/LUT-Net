import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,10,2,1,0),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(10,20,2,1,0),
            nn.ReLU())
        self.fc1=nn.Sequential(
            nn.Linear(20*12,1),
            nn.Sigmoid())
    def forward(self,x):
        x=x.view(-1,1,4,8)
        x=self.conv2(self.conv1(x))
        #print(x.size())
        x=x.view(-1,20*12)
        x=self.fc1(x)
        return x


class FC1(nn.Module):
    def __init__(self,nInput, activate, weight):
        super(FC1, self).__init__()
        self.nInput=nInput

        self.fc1 = nn.Linear(self.nInput, self.nInput*2)
        self.fc2 = nn.Linear(self.nInput*2, self.nInput)
        self.fc3 = nn.Linear(self.nInput, self.nInput//2)
        self.fc4 = nn.Linear(self.nInput//2, 1)
        self.weight = weight
        if activate == 'sigmoid':
            self.activate = nn.Sigmoid()
        else:
            self.activate = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        #print(x.size())
        x1 = self.activate(self.weight*self.dropout(self.fc1(x)))
        x2 = self.activate(self.weight*self.dropout(self.fc2(x1)))
        x3 = self.activate(self.weight*self.fc3(x2))
        x4 = self.sigmoid(self.weight*self.fc4(x3))
        return x1, x2, x3, x4
