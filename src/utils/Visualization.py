import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore")

'''Visualization for acc'''
def visualization(mode,dataset_name,communication_round,num_clients,family_name,uid_list):
    df_B=pd.read_csv("../../result/{} {}/Basic-Common/result.csv".format(family_name,dataset_name))
    df_C=pd.read_csv("../../result/{} {}/Clustered-Common/result.csv".format(family_name,dataset_name))
    df_M=pd.read_csv("../../result/{} {}/Max-Common/result.csv".format(family_name,dataset_name))
    df_S=pd.read_csv("../../result/{} {}/Standalone/result.csv".format(family_name,dataset_name))
    df_CL=pd.read_csv("../../result/{} {}/Clustered-FL/result.csv".format(family_name,dataset_name))
    model_name={
        "VGG":["VGG-11","VGG-13","VGG-16","VGG-19"],
        "ResNet":["ResNet-20","ResNet-32","ResNet-44","ResNet-56"],
        "CharCNN":["CharCNN-3","CharCNN-4","CharCNN-5","CharCNN-6"],
        "VDCNN":["VDCNN-9","VDCNN-17","VDCNN-29","VDCNN-49"]
    }
    '''
    AxesSubplot(0.125,0.25;0.133621x0.5133)
    AxesSubplot(0.285345,0.25;0.133621x0.5133)
    AxesSubplot(0.44569,0.25;0.133621x0.5133)
    AxesSubplot(0.606034,0.25;0.133621x0.5133)
    AxesSubplot(0.766379,0.25;0.133621x0.5133)
    '''
    if mode==0: # the convergence process under different strategy
        fig=plt.figure(figsize=(16,3),dpi=100)
        # visualize the convergence process under Basic-Common strategy
        ax=fig.add_axes([0.125,0.25,0.133621,0.5133])
        draw_plot0(df_S,ax,communication_round,"Standalone",num_clients,family_name,uid_list)
        # visualize the convergence process under Clustered-FL strategy
        ax=fig.add_axes([0.285345,0.25,0.133621,0.5133])
        draw_plot0(df_CL,ax,communication_round,"Clustered-FL",num_clients,family_name,uid_list)
        # visualize the convergence process under Basic-Common strategy
        ax=fig.add_axes([0.44569,0.25,0.133621,0.5133])
        draw_plot0(df_B,ax,communication_round,"Basic-Common",num_clients,family_name,uid_list)
        # visualize the convergence process under Clustered-Common strategy
        ax=fig.add_axes([0.606034,0.25,0.133621,0.5133])
        draw_plot0(df_C,ax,communication_round,"Clustered-Common",num_clients,family_name,uid_list)
        # visualize the convergence process under Max-Common strategy
        ax=fig.add_axes([0.766379,0.25,0.133621,0.5133])
        draw_plot0(df_M,ax,communication_round,"Max-Common",num_clients,family_name,uid_list)

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend( lines, labels,loc="upper center",ncol=4, framealpha=1)
        plt.show()
    '''
    AxesSubplot(0.125,0.25;0.168478x0.5133)
    AxesSubplot(0.327174,0.25;0.168478x0.5133)
    AxesSubplot(0.529348,0.25;0.168478x0.5133)
    AxesSubplot(0.731522,0.25;0.168478x0.5133)
    '''
    if mode==1: # the convergence process under different architecture
        fig=plt.figure(figsize=(13,3),dpi=100)
        # visualize the convergence process under VGG-11
        ax=fig.add_axes([0.125,0.25,0.168478,0.5133])
        draw_plot1(df_S,df_CL,df_B,df_C,df_M,0,ax,communication_round,model_name[family_name][0],num_clients,uid_list)
        # visualize the convergence process under VGG-13
        ax=fig.add_axes([0.327174,0.25,0.168478,0.5133])
        draw_plot1(df_S,df_CL,df_B,df_C,df_M,1,ax,communication_round,model_name[family_name][1],num_clients,uid_list)
        # visualize the convergence process under VGG-13
        ax=fig.add_axes([0.529348,0.25,0.168478,0.5133])
        draw_plot1(df_S,df_CL,df_B,df_C,df_M,2,ax,communication_round,model_name[family_name][2],num_clients,uid_list)
        # visualize the convergence process under VGG-13
        ax=fig.add_axes([0.731522,0.25,0.168478,0.5133])
        draw_plot1(df_S,df_CL,df_B,df_C,df_M,3,ax,communication_round,model_name[family_name][3],num_clients,uid_list)

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend( lines, labels,loc="upper center",ncol=5, framealpha=1)
        plt.show()

    if mode==2:
        fig=plt.figure(figsize=(10,3.2))
        ax=fig.add_axes([0.105,0.1,0.8,0.7])
        draw_bar2(ax)
        ax.set_title(family_name+" on "+dataset_name)

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend( lines, labels,loc="upper center",ncol=5, framealpha=1)
        plt.show()

def draw_plot0(df,ax,communication_round,strategy,num_clients,family_name,uid_list):
    x=[i for i in range(communication_round)]
    v1=list(df.iloc[:,uid_list[0]])
    v2=list(df.iloc[:,uid_list[1]])
    v3=list(df.iloc[:,uid_list[2]])
    v4=list(df.iloc[:,uid_list[3]])
    model_name={
        "VGG":["VGG-11","VGG-13","VGG-16","VGG-19"],
        "ResNet":["ResNet-20","ResNet-32","ResNet-44","ResNet-56"],
        "CharCNN":["CharCNN-3","CharCNN-4","CharCNN-5","CharCNN-6"],
        "VDCNN":["VDCNN-9","VDCNN-17","VDCNN-29","VDCNN-49"]
    }
    ax.plot(x,v1,"midnightblue",label=model_name[family_name][0],linewidth=1.0)
    ax.plot(x,v2,"green",label=model_name[family_name][1],linewidth=1.0)
    ax.plot(x,v3,"dodgerblue",label=model_name[family_name][2],linewidth=1.0)
    ax.plot(x,v4,"gold",label=model_name[family_name][3],linewidth=1.0)
    ax.set_title(strategy)
    ax.set_xticks([i for i in range(0,communication_round+1,10)])
    ax.set_xticklabels([i for i in range(0,communication_round+1,10)])
    ax.set_yticks([i/10 for i in range(int(math.floor(min([min(v1),min(v2),min(v3),min(v4)])*10)),9,1)])
    ax.set_yticklabels([i/10 for i in range(int(math.floor(min([min(v1),min(v2),min(v3),min(v4)])*10)),9,1)])
    ax.grid(linestyle='-.')

def draw_plot1(df_S,df_CL,df_B,df_C,df_M,v,ax,communication_round,architecture,num_clients,uid_list):
    x=[i for i in range(communication_round)]
    vS=list(df_S.iloc[:,uid_list[v]])
    vCL=list(df_CL.iloc[:,uid_list[v]])
    vB=list(df_B.iloc[:,uid_list[v]])
    vC=list(df_C.iloc[:,uid_list[v]])
    vM=list(df_M.iloc[:,uid_list[v]])
    ax.plot(x,vS,"dodgerblue",label="Standalone",linewidth=1.0)
    ax.plot(x,vCL,"lightgreen",label="Clustered-FL",linewidth=1.0)
    ax.plot(x,vB,"midnightblue",label="Basic-Common",linewidth=1.0)
    ax.plot(x,vC,"green",label="Clustered-Common",linewidth=1.0)
    ax.plot(x,vM,"gold",label="Max-Common",linewidth=1.0)


    ax.set_title(architecture)
    ax.set_xticks([i for i in range(0,communication_round+1,10)])
    ax.set_xticklabels([i for i in range(0,communication_round+1,10)])
    ax.set_yticks([i/10 for i in range(int(math.floor(min([min(vB),min(vC),min(vM)])*10)),9,1)])
    ax.set_yticklabels([i/10 for i in range(int(math.floor(min([min(vB),min(vC),min(vM)])*10)),9,1)])
    ax.grid(linestyle='-.')

def draw_bar2(ax):
    ax.set_ylim(0.2,0.95)
    ax.set_xlim(0,86)

    x_Standalone=[7,27,47,67]
    x_Clustered_FL=[10,30,50,70]
    x_Basic_Common=[13,33,53,73]
    x_Clustered_Common=[16,36,56,76]
    x_Max_Common=[19,39,59,79]

    y_Standalone=[0.26,0.69,0.71,0.76]
    y_Clustered_FL=[0.25,0.83,0.74,0.75]
    y_Basic_Common=[0.80,0.86,0.85,0.83]
    y_Clustered_Common=[0.80,0.87,0.82,0.83]
    y_Max_Common=[0.66,0.87,0.85,0.86]

    rects1=ax.bar(x=x_Standalone,height=y_Standalone,width=3,color='dodgerblue',label='Standalone')
    rects2=ax.bar(x=x_Clustered_FL,height=y_Clustered_FL,width=3,color='lightgreen',label='Clustered-FL')
    rects3=ax.bar(x=x_Basic_Common,height=y_Basic_Common,width=3,color='midnightblue',label='Basic-Common')
    rects4=ax.bar(x=x_Clustered_Common,height=y_Clustered_Common,width=3,color='green',label='Clustered-Common')
    rects5=ax.bar(x=x_Max_Common,height=y_Max_Common,width=3,color='gold',label='Max-Common')

    ax.set_xticks(x_Basic_Common)
    ax.set_xticklabels(('V1','V2','V3','V4'))

    def label(ax,rects):
        for rect in rects:
            height=rect.get_height()
            ax.text(rect.get_x(),height+0.01,str(height),fontsize=9)

    label(ax,rects1)
    label(ax,rects2)
    label(ax,rects3)
    label(ax,rects4)
    label(ax,rects5)
    ax.grid(linestyle='-.')