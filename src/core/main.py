import sys
sys.path.append("..")
from utils.Visualization import *
from FlexiFed import *
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    '''Important Parameters'''
    num_clients=8
    family_name="VGG"
    dataset_name="CIFAR-10"
    communication_round=70
    strategy="Clustered-FL"
    stage="Stage1"
    '''Federated Learning'''
    print("Getting Dataset... ")
    train_dataset,test_dataset,train_group,test_group=get_dataset(dataset_name,family_name,num_clients)
    print("Initializing the Parameter Server...")
    Server=ParameterServer(num_clients,family_name,dataset_name,train_group,test_group)
    # print("Start to Federated learning...")
    # Server.ServerGlobalTraining(train_dataset,test_dataset,communication_round,strategy,stage)
    # ''' Visualize the result '''
    visualization(0,stage,dataset_name,communication_round,num_clients,family_name,[0,2,4,6])
    visualization(1,stage,dataset_name,communication_round,num_clients,family_name,[0,2,4,6])
    # '''Test Client'''
    # for uid in range(num_clients):
    #     Server.Clients[uid].model.load_state_dict(
    #         torch.load("../../model/{}/{}/{}/Client{}.pkl".format(stage,strategy,dataset_name,uid),
    #                    map_location=torch.device('cpu')))
    #     test_idx=Server.Clients[uid].test_idx
    #     print("[strategy:{},uid:{}] Accuracy:{}".format(
    #         strategy,uid,Server.Clients[uid].ClientLocalTesting(Server.device,test_dataset,test_idx)))


