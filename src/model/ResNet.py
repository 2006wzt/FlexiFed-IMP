import math
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


class ResNet(nn.Module):
    def __init__(self,block,layers,input_channel,w,h,num_classes):
        super(ResNet,self).__init__()
        self.inplanes = 16
        # Conv1
        self.conv1=nn.Sequential(
            nn.Conv2d(input_channel,self.inplanes,
                      kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
        )
        # Conv2
        self.layer1=self.make_layer(block,16,layers[0])
        # Conv3
        self.layer2=self.make_layer(block,32,layers[1],stride=2)
        # Conv4
        self.layer3=self.make_layer(block,64,layers[2],stride=2)
        # fc
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        '''Initialize the weights of conv layers'''
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        x=self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    def make_layer(self,block,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion=1 # Expansion times of the number of channels
    def __init__(self,in_planes,planes,stride=1,downsample=None):
        '''
        :param in_planes: the input channels of tensor
        :param planes: the output channels of every conv_layer
        :param stride: stride of the kernel
        :param downsample: like Max-pooling
        '''
        super(BasicBlock,self).__init__()
        self.conv=nn.Sequential(
            # Conv1
            nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            # Conv2
            nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(planes)
        )
        # downsample
        self.downsample=downsample
        self.stride=stride
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        residual=x
        out=self.conv(x)
        if self.downsample is not None:
            residual=self.downsample(x)
        out+=residual
        out=self.relu(out)
        return out



def resnet20(input_channel,w,h,num_classes):
    return ResNet(BasicBlock,[3,3,3],input_channel,w,h,num_classes)

def resnet32(input_channel,w,h,num_classes):
    return ResNet(BasicBlock,[5,5,5],input_channel,w,h,num_classes)

def resnet44(input_channel,w,h,num_classes):
    return ResNet(BasicBlock,[7,7,7],input_channel,w,h,num_classes)

def resnet56(input_channel,w,h,num_classes):
    return ResNet(BasicBlock,[9,9,9],input_channel,w,h,num_classes)

# model1=resnet44(3,32,32,10)
# model2=resnet56(3,32,32,10)
# print(model1.state_dict().keys())
# print(model2.state_dict().keys())

# class Bottleneck(nn.Module):
#     expansion=4 # Expansion times of the number of channels
#     def __init__(self,in_planes,planes,stride=1,downsample=None):
#         super(Bottleneck,self).__init__()
#         self.conv=nn.Sequential(
#             # Conv1
#             nn.Conv2d(in_planes,planes,kernel_size=1,bias=False),
#             nn.BatchNorm2d(planes),
#             nn.ReLU(inplace=True),
#             # Conv2
#             nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False),
#             nn.BatchNorm2d(planes),
#             nn.ReLU(inplace=True),
#             # Conv3
#             nn.Conv2d(planes,planes*4,kernel_size=1,bias=False),
#             nn.BatchNorm2d(planes*4)
#         )
#         self.relu=nn.ReLU(inplace=True)
#         self.downsample=downsample
#         self.stride=stride
#     def forward(self,x):
#         residual=x
#         out=self.conv(x)
#         if self.downsample is not None:
#             residual=self.downsample(x)
#         out+=residual
#         out=self.relu(out)
#         return out