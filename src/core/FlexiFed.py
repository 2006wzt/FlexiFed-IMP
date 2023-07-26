import pandas as pd
from Client import *
import warnings
warnings.filterwarnings("ignore")

class ParameterServer():
    '''The parameter server in FL System'''
    '''Initialize the member of Server'''
    def __init__(self,num_clients,family_name,dataset_name,train_group,test_group):
        self.num_clients=num_clients
        self.Clients=get_ClientSet(num_clients,family_name,dataset_name,train_group,test_group)
        self.acc={i:[] for i in range(num_clients)}
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''Clobal Training'''
    def ServerGlobalTraining(self,train_dataset,test_dataset,communication_round,strategy,family_name,data_name):
        localData_len=len(self.Clients[0].train_idx)/10
        start=0
        for epoch in range(communication_round):
            end=start+localData_len
            '''Local Traing'''
            for uid in range(self.num_clients):
                idx_train=self.Clients[uid].train_idx
                idx_batch=idx_train[int(start):int(end)]
                idx_test=self.Clients[uid].test_idx
                model_dict=self.Clients[uid].ClientLocalTraining(self.device,train_dataset,idx_batch)
                acc=self.Clients[uid].ClientLocalTesting(self.device,test_dataset,idx_test)
                self.acc[uid].append(acc)
                print("[epoch:%d,uid:%d] acc:%.5f"%(epoch,uid,acc))
                print()
                self.Clients[uid].model.load_state_dict(model_dict) # maybe it could be commented out
            start=end%int(len(train_dataset)/self.num_clients)
            print("[epoch:{},strategy:{}] Global Training ".format(epoch,strategy))
            print()
            # print(self.Clients[0].model.state_dict()['conv1.0.weight'][0])
            if strategy=="Basic-Common":
                self.Basic_Common(self.Clients,self.Clients.keys())
            elif strategy=="Clustered-Common":
                self.Clustered_Common(self.Clients,self.Clients.keys())
            elif strategy=="Max-Common":
                self.Max_Common(self.Clients,self.Clients.keys())
            elif strategy=="Clustered-FL":
                self.Clustered_FL(self.Clients,self.Clients.keys())
            elif strategy=="Standalone":
                pass
            # print(self.Clients[0].model.state_dict()['conv1.0.weight'][0])
        '''Save the convergence process'''
        print(self.acc)
        df=pd.DataFrame(self.acc)
        df.to_csv("../../result/{} {}/{}/result.csv".format(family_name,data_name,strategy),index=False)
        '''Save the Client's model'''
        for uid in range(self.num_clients):
            torch.save(self.Clients[uid].model.state_dict(),
                       "../../model/{} {}/{}/Client{}.pkl".format(family_name,data_name,strategy,uid))


    '''Basic_Common Strategy'''
    def Basic_Common(self,Clients,uid_list):
        # find the min model in FL System
        min_uid=0
        min_len=100000
        num_clients=len(uid_list)
        for uid in uid_list:
            length=len(Clients[uid].model.state_dict())
            if length<min_len:
                min_len=length
                min_uid=uid
        common_base_layer_dict=Clients[min_uid].model.state_dict()
        common_base_layer_list=list(common_base_layer_dict.keys())
        # match the common base layers
        for uid in uid_list:
            layer_list=list(Clients[uid].model.state_dict().keys())
            layer_dict=Clients[uid].model.state_dict()
            for i in range(len(common_base_layer_list)):
                name_comm=common_base_layer_list[i]
                name=layer_list[i]
                size_comm=common_base_layer_dict[name_comm].size()
                size=layer_dict[name].size()
                #same name and same size->common base layer!
                if (name_comm==name) and (size_comm==size):
                    continue
                else:
                    del common_base_layer_list[i:]
                    break
        # Aggregating the common base layer
        for name in common_base_layer_list:
            weight_comm=torch.zeros(common_base_layer_dict[name].size()).to(self.device)
            for uid in uid_list:
                weight_comm+=Clients[uid].model.state_dict()[name]
            weight_comm/=num_clients
            for uid in uid_list:
                model_dict=Clients[uid].model.state_dict()
                model_dict[name]=weight_comm
                Clients[uid].model.load_state_dict(model_dict)
        return common_base_layer_list

    '''Clustered-Common Strategy'''
    def Clustered_Common(self,Clients,uid_list):
        common_base_layer_list=self.Basic_Common(Clients,uid_list)
        group_dict={}   # store the clients with same architecture
        arch_dict={}    # store the architecture of arch_len
        # split the clients to groups according to the architecture
        for uid in uid_list:
            arch_len=len(Clients[uid].model.state_dict().keys())
            if arch_len not in group_dict.keys():
                group_dict[arch_len]=[uid]
                arch_dict[arch_len]=Clients[uid].model.state_dict().keys()
            else:
                group_dict[arch_len].append(uid)
        # Aggregating the personal layer for every group
        for arch_len in group_dict.keys():
            group=group_dict[arch_len]
            personal_layer_list=[name for name in arch_dict[arch_len] if name not in common_base_layer_list]
            personal_layer_dict=Clients[group[0]].model.state_dict()
            for name in personal_layer_list:
                weight_comm=torch.zeros(personal_layer_dict[name].size()).to(self.device)
                for uid in group:
                    weight_comm+=Clients[uid].model.state_dict()[name]
                weight_comm/=len(group)
                for uid in group:
                    model_dict=Clients[uid].model.state_dict()
                    model_dict[name]=weight_comm
                    Clients[uid].model.load_state_dict(model_dict)

    '''Max-Common Strategy'''
    def Max_Common(self,Clients,uid_list):
        if len(uid_list)==1 or len(uid_list)==0:
            return Clients
        common_base_layer_list=self.Basic_Common(Clients,uid_list)
        group_dict={}
        # split the clients to groups according to the common base layer in personal layer
        for uid in uid_list:
            architecture=Clients[uid].model.state_dict().keys()
            personal_layer_list=[name for name in architecture if name not in common_base_layer_list]
            if len(personal_layer_list)==0:
                continue
            if personal_layer_list[0] not in group_dict.keys():
                group_dict[personal_layer_list[0]]=[uid]
            else:
                group_dict[personal_layer_list[0]].append(uid)
        # Aggregating the common base layer for each group
        for layer in group_dict.keys():
            self.Max_Common(Clients,group_dict[layer])
        return

    '''Clustered-FL'''
    def Clustered_FL(self,Clients,uid_list):
        group_dict={}   # store the clients with same architecture
        arch_dict={}    # store the architecture of arch_len
        # split the clients to groups according to the architecture
        for uid in uid_list:
            arch_len=len(Clients[uid].model.state_dict().keys())
            if arch_len not in group_dict.keys():
                group_dict[arch_len]=[uid]
                arch_dict[arch_len]=Clients[uid].model.state_dict().keys()
            else:
                group_dict[arch_len].append(uid)
        # Aggregating the personal layer for every group
        for arch_len in group_dict.keys():
            group=group_dict[arch_len]
            personal_layer_list=arch_dict[arch_len]
            personal_layer_dict=Clients[group[0]].model.state_dict()
            for name in personal_layer_list:
                weight_comm=torch.zeros(personal_layer_dict[name].size()).to(self.device)
                for uid in group:
                    weight_comm+=Clients[uid].model.state_dict()[name]
                weight_comm/=len(group)
                for uid in group:
                    model_dict=Clients[uid].model.state_dict()
                    model_dict[name]=weight_comm
                    Clients[uid].model.load_state_dict(model_dict)
