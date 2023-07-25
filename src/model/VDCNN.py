import math
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import warnings
warnings.filterwarnings("ignore")

class VDCNN(nn.Module):
    def __init__(self,block,layers,m=69,l0=1024,num_classes=4):
        super(VDCNN, self).__init__()
        self.embedding=nn.Embedding(m,16,padding_idx=0)
        self.conv0=nn.Sequential(
            nn.Conv1d(16,64,kernel_size=3,padding=1),
            ConvBlock(64,64)
        )
        # layer1
        self.layer1=self.make_layer(block,64,layers[0])
        self.pooling1=nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv1=ConvBlock(64,128,downsample=nn.Sequential(
            nn.Conv1d(64,128,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm1d(128),
        ))
        # layer2
        self.layer2=self.make_layer(block,128,layers[1])
        self.pooling2=nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2=ConvBlock(128,256,downsample=nn.Sequential(
            nn.Conv1d(128,256,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm1d(256),
        ))
        # layer3
        self.layer3=self.make_layer(block,256,layers[2])
        self.pooling3=nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3=ConvBlock(256,512,downsample=nn.Sequential(
            nn.Conv1d(256,512,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm1d(512),
        ))
        # layer4
        self.layer4=self.make_layer(block,512,layers[3])
        self.pooling4=nn.AdaptiveMaxPool1d(8)
        # full connection
        self.fc=nn.Sequential(
            nn.Linear(4096,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048,num_classes),
        )
        '''Initialize the weights of conv layers'''
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


    def forward(self,x):
        x=self.embedding(x).transpose(1,2)
        x=self.conv0(x)

        x=self.layer1(x)
        x=self.pooling1(x)
        x=self.conv1(x)

        x=self.layer2(x)
        x=self.pooling2(x)
        x=self.conv2(x)

        x=self.layer3(x)
        x=self.pooling3(x)
        x=self.conv3(x)

        x=self.layer4(x)
        x=self.pooling4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def make_layer(self,block,planes,blocks):
        layers=[]
        for i in range(blocks):
            layers.append(block(planes,planes))
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(self,in_planes,planes,downsample=None):
        super(ConvBlock, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(in_planes,planes,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes,planes,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm1d(planes),
        )
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self,x):
        residual=x
        out=self.conv(x)
        if self.downsample is not None:
            residual=self.downsample(x)
        out+=residual
        out=self.relu(out)
        return out

def vdcnn9(m,l0,num_classes):
    return VDCNN(ConvBlock,[0,0,0,0],m,l0,num_classes)

def vdcnn17(m,l0,num_classes):
    return VDCNN(ConvBlock,[1,1,1,1],m,l0,num_classes)

def vdcnn29(m,l0,num_classes):
    return VDCNN(ConvBlock,[4,4,1,1],m,l0,num_classes)

def vdcnn49(m,l0,num_classes):
    return VDCNN(ConvBlock,[7,7,4,2],m,l0,num_classes)

# model1=vdcnn9(69,1024,4)
# model2=vdcnn17(69,1024,4)
# x=torch.IntTensor(torch.zeros((64,1024)).numpy())
# print(model1.forward(x).size())