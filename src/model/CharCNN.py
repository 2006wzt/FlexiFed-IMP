import math
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


class CharCNN(nn.Module):
    def __init__(self,version,m=70,l0=1014,num_classes=4):
        super(CharCNN,self).__init__()
        self.embedding=nn.Embedding(m+1,m,padding_idx=0)
        self.embedding.weight.data[1:].copy_(torch.eye(m))
        self.embedding.weight.requires_grad=False
        self.conv1=nn.Sequential(
            nn.Conv1d(m,256,kernel_size=7),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(256,256,kernel_size=7),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3),
        )
        self.size=((l0-6)/3-6)/3
        self.version=version
        if version>=4:
            self.layer1=nn.Sequential(
                nn.Conv1d(256,256,kernel_size=3),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )
            self.size-=2
        if version>=5:
            self.layer2=nn.Sequential(
                nn.Conv1d(256,256,kernel_size=3),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )
            self.size-=2
        if version>=6:
            self.layer3=nn.Sequential(
                nn.Conv1d(256,256,kernel_size=3),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )
            self.size-=2
        self.conv2=nn.Sequential(
            nn.Conv1d(256,256,kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3),
        )
        self.size=(self.size-2)/3
        self.fc=nn.Sequential(
            nn.Linear(int(self.size)*256,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024,num_classes),
        )
        '''Initialize the weight of conv layers'''
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                m.weight.data.normal_(0,0.05)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(0,0.05)

    def forward(self,x):
        x=self.embedding(x).transpose(1,2)
        x=self.conv1(x)
        if self.version>=4:
            x=self.layer1(x)
        if self.version>=5:
            x=self.layer2(x)
        if self.version>=6:
            x=self.layer3(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

def charcnn3(m,l0,num_classes):
    return CharCNN(3,m,l0,num_classes)

def charcnn4(m,l0,num_classes):
    return CharCNN(4,m,l0,num_classes)

def charcnn5(m,l0,num_classes):
    return CharCNN(5,m,l0,num_classes)

def charcnn6(m,l0,num_classes):
    return CharCNN(6,m,l0,num_classes)

# model1=charcnn5()
# model2=charcnn6()
# print(model1.state_dict().keys())
# print(model2.state_dict().keys())
