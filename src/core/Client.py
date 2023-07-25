import sys
sys.path.append("..")

from utils.Dataset import *
from model.VGG import *
from model.ResNet import *
from model.CharCNN import *
from model.VDCNN import *
from torch.utils.data import DataLoader
from random import shuffle
import warnings
warnings.filterwarnings("ignore")

class Client():
    '''A Client in FL System'''
    '''Initialize the member of Client'''
    def __init__(self,uid,model,dataset_name,train_idx,test_idx):
        self.uid=uid
        self.model=model
        self.dataset_name=dataset_name
        self.train_idx=list(train_idx)
        shuffle(self.train_idx) # necessary!!!
        self.test_idx=list(test_idx)
    '''Local Training 10 epoch'''
    def ClientLocalTraining(self,device,train_dataset,train_idx):
        # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        loss_func=nn.CrossEntropyLoss()
        optimizer=torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # optimizer=torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        if self.dataset_name=="Speech-Commands":
            train_loader=DataLoader(DatasetSplitSpeech(train_dataset,train_idx),batch_size=64,shuffle=True)
        else:
            train_loader=DataLoader(DatasetSplit(train_dataset,train_idx),batch_size=64,shuffle=True)
        for epoch in range(10):
            print("[client uid:%d,epoch:%d] Local Training..."%(self.uid,epoch))
            for iter_,(xb,yb) in enumerate(train_loader,0):
                xb,yb=xb.to(device),yb.to(device)
                optimizer.zero_grad()
                pred=self.model(xb)
                loss=loss_func(pred,yb)
                loss.backward()
                optimizer.step()
        return self.model.state_dict()
    '''Test for Accuracy'''
    def ClientLocalTesting(self,device,test_dataset,test_idx):
        self.model.to(device)
        self.model.eval()
        total,correct=0.0,0.0
        if self.dataset_name=="Speech-Commands":
            test_loader=DataLoader(DatasetSplitSpeech(test_dataset,test_idx),batch_size=64,shuffle=True)
        else:
            test_loader=DataLoader(DatasetSplit(test_dataset,test_idx),batch_size=64,shuffle=True)
        for iter_,(xb,yb) in enumerate(test_loader,0):
            xb,yb=xb.to(device),yb.to(device)
            pred=self.model(xb).argmax(dim=1)
            total+=xb.size(0)
            correct+=torch.eq(pred,yb).sum().item()
        acc=correct*1.0/total
        return acc

def get_ClientSet(num_clients,family_name,dataset_name,train_group,test_group):
    client_dict={}
    input_channel,w,h=0,0,0
    num_classes=0
    if dataset_name=="CIFAR-10" or dataset_name=="CINIC-10":
        input_channel=3
        w,h=32,32
        num_classes=10
    elif dataset_name=="Speech-Commands":
        input_channel=1
        w,h=32,32
        num_classes=12
    if family_name=="VGG":
        num_arch=int(num_clients/4)
        for uid in range(num_clients):
            if math.floor(uid/num_arch)==0:
                client_dict[uid]=Client(uid,vgg11_bn(input_channel,w,h,num_classes),
                                        dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==1:
                client_dict[uid]=Client(uid,vgg13_bn(input_channel,w,h,num_classes),
                                        dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==2:
                client_dict[uid]=Client(uid,vgg16_bn(input_channel,w,h,num_classes),
                                        dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==3:
                client_dict[uid]=Client(uid,vgg19_bn(input_channel,w,h,num_classes),
                                        dataset_name,train_group[uid],test_group[uid])
        return client_dict
    elif family_name=="ResNet":
        num_arch=int(num_clients/4)
        for uid in range(num_clients):
            if math.floor(uid/num_arch)==0:
                client_dict[uid]=Client(uid,resnet20(input_channel,w,h,num_classes),
                                        dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==1:
                client_dict[uid]=Client(uid,resnet32(input_channel,w,h,num_classes),
                                        dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==2:
                client_dict[uid]=Client(uid,resnet44(input_channel,w,h,num_classes),
                                        dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==3:
                client_dict[uid]=Client(uid,resnet56(input_channel,w,h,num_classes),
                                        dataset_name,train_group[uid],test_group[uid])
        return client_dict
    elif family_name=="CharCNN":
        num_arch=int(num_clients/4)
        for uid in range(num_clients):
            if math.floor(uid/num_arch)==0:
                client_dict[uid]=Client(uid,charcnn3(70,1014,4),dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==1:
                client_dict[uid]=Client(uid,charcnn4(70,1014,4),dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==2:
                client_dict[uid]=Client(uid,charcnn5(70,1014,4),dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==3:
                client_dict[uid]=Client(uid,charcnn6(70,1014,4),dataset_name,train_group[uid],test_group[uid])
        return client_dict
    elif family_name=="VDCNN":
        num_arch=int(num_clients/4)
        for uid in range(num_clients):
            if math.floor(uid/num_arch)==0:
                client_dict[uid]=Client(uid,vdcnn9(69,1024,4),dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==1:
                client_dict[uid]=Client(uid,vdcnn17(69,1024,4),dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==2:
                client_dict[uid]=Client(uid,vdcnn29(69,1024,4),dataset_name,train_group[uid],test_group[uid])
            elif math.floor(uid/num_arch)==3:
                client_dict[uid]=Client(uid,vdcnn49(69,1024,4),dataset_name,train_group[uid],test_group[uid])
        return client_dict

# num_clients=8
# train_dataset,test_dataset,train_group,test_group=get_dataset("CIFAR-10",num_clients)
# dict=get_ClientSet(num_clients,"VGG",train_group,test_group)
# print(dict[5].model)