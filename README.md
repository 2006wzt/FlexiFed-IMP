# FlexiFed-IMP

Implementation of paper - [FlexiFed: Personalized Federated Learning for Edge Clients with Heterogeneous Model Architectures](https://dl.acm.org/doi/10.1145/3543507.3583347)

## Background

Federated Learning (FL) allows edge mobile and Web-of-Things devices to train a shared global Machine Learning (ML) model under the orchestration of a center parameter server, which has tackled the challenge of privacy issue and performance issue. But the existing FL schemes are mainly concerned with data heterogeneity and they cannot accommodate architecture heterogeneity - clients running ML models with different architectures cannot train a ML model collaboratively under these schemes.

In this paper, experiments are conducted with four widely-used ML models on four public datasets, which demonstrate the usefulness of FlexiFed. There are three aggregation strategies under the frame of the FlexiFed and I will present the processes and results of Implementation of the paper in this repository.

## Prerequisites

- Python 3.8 (ubuntu20.04)

- Pytorch 2.0.0

- Cuda 11.8

- GPU RTX 4090

- Necessary lib

  ```bash
  pip install numpy
  pip install pandas
  pip install librosa==0.9.1
  ```

The environment for running the project is provided by [AutoDL](https://www.autodl.com/home).

## Structure

The structure of the project is shown below :

```bash
├─src
│  ├─model     
|  |  ├─VGG.py
|  |  ├─ResNet.py
|  |  ├─CharCNN.py
|  |  └─VDCNN.py
|  |
|  ├─utils
|  |  ├─Dataset.py
|  |  └─Visualization.py
|  |
|  └─core
|     ├─Client.py
|     ├─FlexiFed.py
|     └─main.py
|
├─dataset
│  ├─CIFAR-10     
|  ├─CINIC-10
|  ├─Speech-Commands
|  └─AG-News
|
└─README.md
```

- src/model : 

  > - VGG.py : There are four models in VGG-family - VGG-11, VGG-13, VGG-16, VGG-19
  > - ResNet.py : There are four models in ResNet-family - ResNet-20, VGG-32, VGG-44, VGG-56
  > - CharCNN.py : There are four models in CharCNN-family - CharCNN-3, CharCNN-4, CharCNN-5, CharCNN-6
  > - VDCNN.py : There are four models inVDCNN-family - VDCNN-9, VDCNN-17, VDCNN-29, VDCNN-49

  Use the functions provided by each pythonfile to create the model you want, e.g. :

  ```python
  model1=vgg11_bn(input_channel=3,w=32,h=32,num_classes=10)
  model2=resnet20(input_channel=3,w=32,h=32,num_classes=10)
  model3=charcnn3(m=70,l0=1014,num_classes=4)
  model4=vdcnn9(m=69,l0=1024,num_classes=4)
  ```

- src/utils :

  > - Dataset.py : There are some necessary functions to get the dataset (`get_dataset`)  and split the dataset by index (`get_idx_dict`) . The relevant dataset includes : CIFAR-10, CINIC-10, Speech-Commands, AG-News
  > - Visualization : There are functions to visualize the result of the FL schemes and two figures will be created : the convergence process of models in the same family with different version under the same aggregation strategy, the convergence process of models in the same family with same version under different aggregation strategy

- src/core:

  > - Clients.py : There is an important class to simulate the clients in edge, which has the ability to train and test locally
  > - FlexiFed.py : The FlexiFed frame named ParamserServer, which has a client-set to simulate the clients in the FL System, has three aggregation strategies (Basic-Common, Clustered-Common, Max-Common) and a function to train clients globally
  > - main.py : The main function to run the FL System, you can run different system by setting the parameters : num_clients (the number of the clients in FL System), family_name (the model family), dataset_name, communication_round (the rounds of global training), strategy (the aggregation strategy)

## Datasets

Four datasets are used to train the model :

- CIFAR-10 : A dataset used for image classification, which has 60,000 32x32 RGB images in 10 classes (train : test = 5 : 1) .

  You can use the pytorch to download the CIFAR-10 dataset directly, like :

  ```python
  from torchvision import datasets
  train_dataset = datasets.CIFAR10(path+"/train",train=True,download=False,transform=data_transforms['train'])
  test_dataset = datasets.CIFAR10(path+"/test",train=False,download=False,transform=data_transforms['test'])
  ```

  We use this dataset to train VGG-family and ResNet-family.

- CINIC-10 : A dataset used for image classification, which has 270,000 32x32 RGB images in 10 classes (train : test = 2 : 1 ) .

  For this dataset, I combine the training samples and validation samples to train the model, you can [use this link](https://datashare.ed.ac.uk/handle/10283/3192) to download the dataset and merage train-valid locally. To create the dataset in pytorch, you can use the ImageFolder, like :

  ```python
  from torchvision import datasets
  train_dataset = datasets.ImageFolder(path+"/train",transform=data_transforms['train'])
  test_dataset = datasets.ImageFolder(path+"/test",transform=data_transforms['test'])
  ```

  We use this dataset to train VGG-family and ResNet-family.

- Speech-Commands : A dataset used for speech recognition, which has 65,000 one-second-long audio clips of different keywords. We just care about the classes include : 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', so that we will set the label of other classes to '\_unknown\_'. Note that there is also a class named '\_silence\_', I split the audio in \_background\_noise\_ to one-second-long clips and add some waveforms initialized by zero to constitute the '\_silence\_' data in the train dataset.

  You can [use this link](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz) to download the train dataset and [use this link](http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz) to download the test dataset. Note that the version of the Speech-Commands dataset I used is v0.01. You will get a 1x16,000 array after loading an audio and I will transform this array to a 1x32x32 matrix by some complicated operations ( you can read the code of `src/utils/Dataset.py` to get some details) . Finally, we can load the dataset by the class we design, like :

  ```python
  train_dataset=Speech_CommandsDataset(path+"/train",data_transforms['train'])
  test_dataset=Speech_CommandsDataset(path+"/test",data_transforms['test'])
  ```

  We use this dataset to train VGG-family and ResNet-family.

- AG-News : A dataset used for text classification, which has 120,000 training samples and 7,600 testing samples. The data is stored in train.csv and test.csv, we will read the columns to get enough characters for our model.

  You can [use this link](https://github.com/antonikow/Vanilla-RNN/tree/main/data/datasets/AG_NEWS) to download the dataset. Let m be the length of the alphabet of the model, l0 be the number of the characters used by the model (e.g. CharCNN - m=70,l0=1014; VDCNN - m=69,l0=1024) , and we will get a mxl0 matrix after loading the news. You can use the AG_NewsDataset ( you can read the code of `src/utils/Dataset.py` to get some details ) to load the dataset, like :

  ```python
  # CharCNN
  train_dataset=AG_NewsDataset(path+"/train/train.csv",1014,alphabet)
  test_dataset=AG_NewsDataset(path+"/test/test.csv",1014,alphabet)
  # VDCNN
  train_dataset=AG_NewsDataset(path+"/train/train.csv",1024,alphabet)
  test_dataset=AG_NewsDataset(path+"/test/test.csv",1024,alphabet)
  ```

  We use this dataset to train CharCNN-family and VDCNN-family.

## Model

Four model familes are used to implement the experiment :

- VGG-family : Visual Geometry Group is an architecture of deep neural network (nn) , it performed very well on ILSVRC 2014 and proved that increasing the depth of the network can affect the final performance of the network to a certain extent.

  We use the four versions of VGG-family : VGG-11, VGG-13, VGG-16, VGG-19, and the architectures of the nn are shown below :

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/arch/VGG-family.png" alt="VGG-family.png" width=661 height=364/>
  </div>

  We can see that there is only one common base layer between different versions of VGG-family, so that we can expect that the difference between different aggregation strategies in VGG model may be obvious.

- ResNet-family : ResNet is proposed mainly to tackle the degradation problem in deep nn and realize the proportional relationship between the depth of the network and model accuracy. The nn versions we use are stacks of BasicBlock, whose architecture is shown below (BasicBlock(planes) ) :

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/arch/BasicBlock.png" alt="BasicBlock.png" width=271 height=196 />
  </div>

  Note that the downsample layer in BasicBlock may not be needed sometimes. We use the four versions of ResNet-family : ResNet-20, ResNet-32, ResNet-44, ResNet-56, and the architectures of the nn are shown below :

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/arch/ResNet-family.png" alt="ResNet-family.png" width=521 height=354 />
  </div>


  The layer's parameters of ResNet-family include : [3,3,3], [5,5,5], [7,7,7], [9,9,9].

- CharCNN : Character-level convolutional neural networks proves that convolutional neural networks can also implement text classification with finer granularity.

  We use the four versions of CharCNN-family : CharCNN-3, CharCNN-4, CharCNN-5, CharCNN-6, and the architectures of the nn are shown below :

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/arch/CharCNN-family.png" alt="CharCNN-family.png" width=509 height=317 />
  </div>

  In my mind, the CharCNN models designed in this project are easy to converge to local minimun on the AG_News dataset. The convergence process will be unexpected sometimes.

- VDCNN : Very Deep Convolutional neural network is similar to ResNet, allowing deeper networks to bring higher accuracy by computing residuals. The nn versions we use are stacks of ConvBlock, whose architecture is shown below (ConvBlock(planes) ) :

  <div align=center>
  <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/arch/ConvBlock.png" alt="ConvBlock.png" width=271 height=196 />
  </div>

  Note that the downsample layer in ConvBlock may not be needed sometimes. We use the four versions of VDCNN-family : VDCNN-9, VDCNN-17, VDCNN-29, VDCNN-49, and the architectures of the nn are shown below :

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/arch/VDCNN-family.png" alt="VDCNN-family.png" weight=497 height=404 />
  </div>
  
  The layer's parameters of VDCNN-family include : [0,0,0,0], [1,1,1,1], [4,4,1,1], [7,7,4,2]. According to my experimental results, the convergence process of VDCNN is oscillating, and the accuracy rate is not rising steadily. 

## Parameter Setting

- Optimizer : torch.optim.SGD
- Learning Rate : 0.01
- Momentum : 0.9
- Weight_decay : 5e-4
- num_clients : 8
- communication_round : 70

The optimizer for local training is SGD and its parameters (lr, momentum, weight_decay) are shown above. We set the number of clients in the FL system to 8, and each two clients share a version of the model-family (e.g. [0,1] - VGG-11, [2,3] - VGG-13, [4,5] - VGG-16, [6,7] - VGG-19) . Every client has a uid to identify them uniquely and the parameter server will use the uid to access these clients in order to aggregate the model of these clients.

## Training Tips

Here are some important tips to help you implement the global training correctly :

- We will divide the dataset equally according to the number of clients in order to simulate that the data from different clients are personalized i.e. data heterogeneity.

- We will divide each client's dataset into 10 batches and select the batch in trun for local training in order to simulate that the data for each local training is not necessarily the same.

- To promise the data diversity of a batch for local training, the shuffle operation before dividing the dataset is necessary, e.g. 

  The label of a batch before shuffle :

  ```python
  tensor([7, 0, 7, 0, 7, 0, 7, 7, 0, 7, 0, 7, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
          0, 0, 0, 0, 7, 0, 7, 7, 7, 7, 7, 7, 0, 0, 7, 0, 7, 7, 0, 0, 7, 0, 0, 0,
          0, 0, 7, 7, 7, 0, 7, 0, 7, 0, 0, 7, 0, 7, 0, 7])
  ```

  The label of a batch after shuffle :

  ```python
  tensor([6, 4, 7, 7, 7, 6, 2, 0, 7, 0, 5, 9, 4, 0, 3, 7, 6, 3, 9, 0, 5, 7, 7, 9,
          5, 5, 3, 3, 0, 6, 1, 7, 9, 5, 3, 7, 0, 7, 1, 0, 4, 6, 6, 3, 2, 9, 4, 4,
          0, 2, 4, 8, 3, 3, 9, 1, 3, 1, 6, 2, 4, 8, 5, 7])
  ```

- Because the purpose of this project is mainly to prove the effectiveness of FlexiFed aggregation strategy, I do not spend too much attention on model design and data processing. Readers can further improve the accuracy of the involved model through some Data Augmentation operations.

## Results

I will show you the results of federated learning for models with architecture heterogeneity on different datasets. Each result includes two pictures ( the convergence process under different strategy and different model ) and one table ( the final accuracy of the model trained by the FL system).

We assume that the number of clients in FL system is n, the accuracy of version v model is Acc_v, then there is :


$$
Acc_v=\frac4n\sum_{i=1}^{n/4}Acc_{v-i}
$$

- VGG-family on CIFAR-10 :

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_CIFAR_10_1.png"/>
  </div>

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_CIFAR_10_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.6936 | 0.7480 | 0.7084 | 0.7364 |
  |   Clustered-FL   | 0.7848 | 0.8068 | 0.7960 | 0.8152 |
  |   Basic-Common   | 0.7136 | 0.7368 | 0.7184 | 0.7152 |
  | Clustered-Common | 0.7560 | 0.8000 | 0.7968 | 0.7892 |
  |    Max-Common    | 0.7680 | 0.8428 | 0.8640 | 0.8428 |
  
- VGG-family on CINIC-10 :

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_CINIC_10_1.png"/>
  </div>

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_CINIC_10_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.6319 | 0.6730 | 0.6434 | 0.6561 |
  |   Clustered-FL   | 0.7111 | 0.7404 | 0.7292 | 0.7331 |
  |   Basic-Common   | 0.6129 | 0.6619 | 0.6379 | 0.6225 |
  | Clustered-Common | 0.6895 | 0.7317 | 0.7420 | 0.7228 |
  |    Max-Common    | 0.6772 | 0.7772 | 0.7892 | 0.7874 |

- VGG-family on Speech Commands :

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_Speech_Commands_1.png"/>
  </div>

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/VGG_Speech_Commands_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.9207 | 0.9385 | 0.9077 | 0.9184 |
  |   Clustered-FL   | 0.9516 | 0.9468 | 0.9681 | 0.9575 |
  |   Basic-Common   | 0.9219 | 0.9385 | 0.9055 | 0.9126 |
  | Clustered-Common | 0.9373 | 0.9563 | 0.9587 | 0.9598 |
  |    Max-Common    | 0.9327 | 0.9787 | 0.9764 | 0.9764 |

- ResNet-family on CIFAR-10 :

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_CIFAR_10_1.png"/>
  </div>

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_CIFAR_10_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.6764 | 0.7044 | 0.6884 | 0.7052 |
  |   Clustered-FL   | 0.7824 | 0.7604 | 0.7796 | 0.7624 |
  |   Basic-Common   | 0.7156 | 0.7164 | 0.7000 | 0.6908 |
  | Clustered-Common | 0.7856 | 0.7672 | 0.7888 | 0.7752 |
  |    Max-Common    | 0.7828 | 0.7944 | 0.7944 | 0.7956 |

- ResNet-family on CINIC-10 :

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_CINIC_10_1.png"/>
  </div>

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_CINIC_10_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3   |   V4   |
  | :--------------: | :----: | :----: | :----: | :----: |
  |    Standalone    | 0.6190 | 0.6470 | 0.6402 | 0.6360 |
  |   Clustered-FL   | 0.7014 | 0.7067 | 0.7156 | 0.7162 |
  |   Basic-Common   | 0.6525 | 0.6385 | 0.6528 | 0.6539 |
  | Clustered-Common | 0.7066 | 0.7119 | 0.7300 | 0.7249 |
  |    Max-Common    | 0.7233 | 0.7252 | 0.7283 | 0.7281 |

- ResNet-family on Speech Commands :

  <div align=center>
    <img src="http://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_Speech_Commands_1.png"/>
  </div>

  <div align=center>
    <img src="https://raw.githubusercontent.com/2006wzt/FlexiFed-IMP/master/images/result/ResNet_Speech_Commands_2.png"/>
  </div>

  |      Scheme      |   V1   |   V2   |   V3    |   V4   |
  | :--------------: | :----: | :----: | :-----: | :----: |
  |    Standalone    | 0.9101 | 0.9160 | 0.8723  | 0.9031 |
  |   Clustered-FL   | 0.9326 | 0.9480 | 0.9350  | 0.9480 |
  |   Basic-Common   | 0.9031 | 0.9054 | 0.9243  | 0.9291 |
  | Clustered-Common | 0.9551 | 0.9445 | 0.94325 | 0.9303 |
  |    Max-Common    | 0.9374 | 0.9563 | 0.9610  | 0.9433 |

- CharCNN-family on AG News :

  

- VDCNN-family on AG News :



## References

Here are some repositories and articles that I refer to during the implementation of the project :

- Project Structure : 

  [fio1982/FlexiFed (github.com)](https://github.com/fio1982/FlexiFed)

- The architecture of ResNet : 

  [pytorch_resnet_cifar10: Proper implementation of ResNet-s for CIFAR10/100 in pytorch that matches description of the original paper.](https://github.com/akamaster/pytorch_resnet_cifar10)

- Loading of Speech Commands dataset : 

  [pytorch-speech-commands: Speech commands recognition with PyTorch | Kaggle 10th place solution in TensorFlow Speech Recognition Challenge ](https://github.com/tugstugi/pytorch-speech-commands)

- The architecture of CharCNN : 

  [charcnn-classification: Character-level Convolutional Networks for Text Classification in Pytorch](https://github.com/Sandeep42/charcnn-classification)

- The architecture of VDCNN : 

  [Very-deep-cnn-pytorch: Very deep CNN for text classification](https://github.com/uvipen/Very-deep-cnn-pytorch)

- Loading of AG News dataset : 

  [pytorch-char-cnn-text-classification: A pytorch implementation of the paper "Character-level Convolutional Networks for Text Classification"](https://github.com/cswangjiawei/pytorch-char-cnn-text-classification)

