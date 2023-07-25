import math
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

'''the architecture of conv layer'''
cfg={
    'A':[64,    'M',128,    'M',256,256,        'M',512,512,        'M',512,512,        'M'],
    'B':[64,64, 'M',128,128,'M',256,256,        'M',512,512,        'M',512,512,        'M'],
    'C':[64,64, 'M',128,128,'M',256,256,256,    'M',512,512,512,    'M',512,512,512,    'M'],
    'D':[64,64, 'M',128,128,'M',256,256,256,    'M',512,512,512,512,'M',512,512,512,512,'M'],
}

class VGG(nn.Module):
    def __init__(self,conv,w,h,num_classes):
        '''
        :param conv: the conv layers of the nn
        :param input_size: the size of the data
        :param num_classes: the num of the classes
        '''
        super(VGG, self).__init__()
        '''Design the architecture of nn'''
        self.conv=conv
        self.fc=nn.Sequential(
            nn.Linear(int(512*w*h/32/32),4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
        )
        '''Initialize the weights of conv layers'''
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                m.bias.data.zero_()

    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

def make_layers(cfg,input_channel,batch_norm=False):
    '''
    :param cfg: the architecture of conv layer
    :param input_channel: the num of input channel
    :param batch_norm: whether to normalize before activation
    :return: conv layers
    '''
    layers=[]
    for l in cfg:
        if l=='M':  # add maxpooling layer
            layers+=[nn.MaxPool2d(kernel_size=2, stride=2)]
        else:   # add conv layer
            layers+=[nn.Conv2d(input_channel,l,kernel_size=3,padding=1)]
            if batch_norm:
                layers+=[nn.BatchNorm2d(l)]
            layers+=[nn.ReLU(inplace=True)]
            input_channel=l
    return nn.Sequential(*layers)

'''Create different model of VGG family'''

'''Create different model of VGG family'''

def vgg11(input_channel,w,h,num_classes):
    return VGG(make_layers(cfg['A'],input_channel),w,h,num_classes)

def vgg11_bn(input_channel,w,h,num_classes):
    return VGG(make_layers(cfg['A'],input_channel,True),w,h,num_classes)

def vgg13(input_channel,w,h,num_classes):
    return VGG(make_layers(cfg['B'],input_channel),w,h,num_classes)

def vgg13_bn(input_channel,w,h,num_classes):
    return VGG(make_layers(cfg['B'],input_channel,True),w,h,num_classes)

def vgg16(input_channel,w,h,num_classes):
    return VGG(make_layers(cfg['C'],input_channel),w,h,num_classes)

def vgg16_bn(input_channel,w,h,num_classes):
    return VGG(make_layers(cfg['C'],input_channel,True),w,h,num_classes)

def vgg19(input_channel,w,h,num_classes):
    return VGG(make_layers(cfg['D'],input_channel),w,h,num_classes)

def vgg19_bn(input_channel,w,h,num_classes):
    return VGG(make_layers(cfg['D'],input_channel,True),w,h,num_classes)

# model1=vgg16_bn().state_dict()
# model2=vgg19_bn().state_dict()
# print(model1.keys())
# print(model2.keys())
# if model1['conv.0.weight'].size()==model2['conv.0.weight'].size():
#     print(list(model1.keys()))
#     print(vgg19_bn())
#     print(model1['conv.1.running_mean'].size())
# a=torch.zeros(model1['conv.0.weight'].size())
# print(a)