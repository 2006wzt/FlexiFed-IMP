import csv
import librosa
import torch
import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

'''split the idx for every client'''
def get_idx_dict(dataset,num_clients):
    '''
    :param dataset: the dataset which will be splited
    :param num_clients: the num of clients in the FL System
    :return: the idx dict for every client
    '''
    num_items=int(len(dataset)/num_clients)
    dict_clients={}
    idxs=[i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_clients[i]=set(np.random.choice(idxs,num_items,replace=False))
        idxs=list(set(idxs)-dict_clients[i])
    return dict_clients
        
'''get the dataset for training'''
def get_dataset(dataset_name,family_name,num_clients):
    dataset_path="../../dataset/"
    if dataset_name=="CIFAR-10":
        path=dataset_path+dataset_name
        mean=(0.4914, 0.4822, 0.4465)
        std=(0.2023, 0.1994, 0.2010)
        data_transforms={
            'train':transforms.Compose([
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
                # normalize:the param is the mean value and standard deviation of the three channel
            ]),
            'test':transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ]),
        }
        train_dataset = datasets.CIFAR10(path+"/train",train=True,download=False,transform=data_transforms['train'])
        test_dataset = datasets.CIFAR10(path+"/test",train=False,download=False,transform=data_transforms['test'])
        train_group=get_idx_dict(train_dataset,num_clients)
        test_group=get_idx_dict(test_dataset,num_clients)
        return train_dataset,test_dataset,train_group,test_group

    elif dataset_name=="CINIC-10":
        path=dataset_path+dataset_name
        mean=(0.4789, 0.4723, 0.4305)
        std=(0.2023, 0.1994, 0.2010)
        data_transforms={
            'train':transforms.Compose([
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
                # normalize:the param is the mean value and standard deviation of the three channel
            ]),
            'test':transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,std=std),
            ]),
        }
        train_dataset = datasets.ImageFolder(path+"/train",transform=data_transforms['train'])
        test_dataset = datasets.ImageFolder(path+"/test",transform=data_transforms['test'])
        train_group=get_idx_dict(train_dataset,num_clients)
        test_group=get_idx_dict(test_dataset,num_clients)
        return train_dataset,test_dataset,train_group,test_group

    elif dataset_name=="Speech-Commands":
        path=dataset_path+dataset_name
        transform1=transforms.Compose([ChangeAmplitude(), FixAudioLength(), ToSTFT(),
                            StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
        transform2=transforms.Compose([ToMelSpectrogramFromSTFT(n_mels=32), DeleteSTFT(),
                            ToTensor('mel_spectrogram', 'input')])
        transform3=transforms.Compose([ToMelSpectrogram(n_mels=32),
                                       ToTensor('mel_spectrogram', 'input')])
        data_transforms={
            'train':transforms.Compose([
                LoadAudio(),
                transform1,
                transform2,
            ]),
            'test':transforms.Compose([
                LoadAudio(),
                FixAudioLength(),
                transform3,
            ]),
        }
        train_dataset=Speech_CommandsDataset(path+"/train",data_transforms['train'])
        test_dataset=Speech_CommandsDataset(path+"/test",data_transforms['test'])
        train_group=get_idx_dict(train_dataset,num_clients)
        test_group=get_idx_dict(test_dataset,num_clients)
        return train_dataset,test_dataset,train_group,test_group

    elif dataset_name=="AG-News":
        path=dataset_path+dataset_name
        if family_name=="CharCNN":
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"
            train_dataset=AG_NewsDataset(path+"/train/train.csv",1014,alphabet)
            test_dataset=AG_NewsDataset(path+"/test/test.csv",1014,alphabet)
            train_group=get_idx_dict(train_dataset,num_clients)
            test_group=get_idx_dict(test_dataset,num_clients)
            return train_dataset,test_dataset,train_group,test_group
        elif family_name=="VDCNN":
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
            train_dataset=AG_NewsDataset(path+"/train/train.csv",1024,alphabet)
            test_dataset=AG_NewsDataset(path+"/test/test.csv",1024,alphabet)
            train_group=get_idx_dict(train_dataset,num_clients)
            test_group=get_idx_dict(test_dataset,num_clients)
            return train_dataset,test_dataset,train_group,test_group




'''split the dataset to feature and label'''
class DatasetSplit(Dataset):
    def __init__(self,dataset,idxs):
        self.dataset=dataset
        self.idxs=list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        feature,label=self.dataset[self.idxs[item]]
        return feature,label

class DatasetSplitSpeech(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        data = self.dataset[self.idxs[item]]
        feature=data['input'].unsqueeze(dim=0)
        label=data['target']
        return feature,label

'''Create the AG_News dataset'''
class AG_NewsDataset(Dataset):
    def __init__(self,path,l0,alphabet):
        super(AG_NewsDataset,self).__init__()
        self.alphabet = alphabet
        self.path=path
        self.l0=l0
        self.data,self.label=self.load_data()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        data=self.data[item]
        data_tensor=torch.zeros(self.l0).long()
        for i,char in enumerate(data):
            if i==self.l0:
                break
            index=self.alphabet.find(char)
            if index!=-1:
                data_tensor[i]=index
        label_tensor=torch.tensor(self.label[item])
        return data_tensor,label_tensor

    def load_data(self):
        data=[]
        label=[]
        with open(self.path,'r') as f:
            csv_reader=csv.reader(f,delimiter=',',quotechar='"')
            for row in csv_reader:
                text=' '.join(row[1:]).lower()
                data.append(text)
                label.append(int(row[0])-1)
        return data,label

'''Create the Speech-Commands dataset'''
class Speech_CommandsDataset(Dataset):
    def __init__(self,folder,transform=None,silence_rate=0.1):
        '''
        :param folder: the path of the dataset
        :param transform: some transform operation
        :param silence_rate: the rate of silence data in the dataset
        '''
        super(Speech_CommandsDataset, self).__init__()
        self.folder=folder
        actual_labels=[d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        labels=['_silence_', '_unknown_', 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
        label_to_idx={labels[i]:i for i in range(len(labels))}
        for l in actual_labels:
            if l not in label_to_idx.keys():
                if l=='_background_noise_':
                    label_to_idx[l]=label_to_idx['_silence_']
                else:
                    label_to_idx[l]=label_to_idx['_unknown_']
        data=[]
        for l in actual_labels:
            d=os.path.join(folder,l)
            target=label_to_idx[l]
            for f in os.listdir(d):
                data.append((os.path.join(d,f),target))
        target = label_to_idx['_silence_']
        data += [('', target)] * int(len(data) * silence_rate)
        self.data=data
        self.transform=transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path,target=self.data[item]
        data={'path':path,'target':target}
        if self.transform!=None:
            data = self.transform(data)
        return data

class LoadAudio(object):    # load the audio according to the path
    def __init__(self,sample_rate=16000):
        self.sample_rate=sample_rate
    def __call__(self, data):
        path=data['path']
        if path:
            samples,sample_rate=librosa.load(path,sr=self.sample_rate)
        else:   # _silence_
            sample_rate = self.sample_rate
            samples = np.zeros(sample_rate, dtype=np.float32)
        data['samples'] = samples
        data['sample_rate'] = sample_rate
        return data

class FixAudioLength(object):   # either pads or truncates an audio into a fixed length
    def __init__(self,time=1):
        self.time=time
    def __call__(self,data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)
        if length < len(samples):
            data['samples'] = samples[:length]
        elif length > len(samples):
            data['samples'] = np.pad(samples, (0, length - len(samples)), "constant")
        return data

class ChangeAmplitude(object):  # change the amplitude randomly
    def __init__(self, amplitude_range=(0.7, 1.1)):
        self.amplitude_range = amplitude_range
    def __call__(self, data):
        if random.random() >= 0.5:
            return data
        data['samples'] = data['samples'] * random.uniform(*self.amplitude_range)
        return data

class ToMelSpectrogram(object):
    def __init__(self, n_mels=32):
        self.n_mels = n_mels
    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        s = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=self.n_mels)
        data['mel_spectrogram'] = librosa.power_to_db(s, ref=np.max)
        return data

class ToSTFT(object): # Applies on an audio the short time fourier transform
    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length
    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        data['n_fft'] = self.n_fft
        data['hop_length'] = self.hop_length
        data['stft'] = librosa.stft(samples, n_fft=self.n_fft, hop_length=self.hop_length)
        data['stft_shape'] = data['stft'].shape
        return data

class StretchAudioOnSTFT(object):   # Stretches an audio on the frequency domain.
    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale
    def __call__(self, data):
        if random.random() >= 0.5:
            return data
        stft = data['stft']
        sample_rate = data['sample_rate']
        hop_length = data['hop_length']
        scale = random.uniform(-self.max_scale, self.max_scale)
        stft_stretch = librosa.core.phase_vocoder(stft, 1+scale, hop_length=hop_length)
        data['stft'] = stft_stretch
        return data

class TimeshiftAudioOnSTFT(object): # A simple timeshift on the frequency domain without multiplying with exp.
    def __init__(self, max_shift=8):
        self.max_shift = max_shift
    def __call__(self, data):
        if random.random() >= 0.5:
            return data
        stft = data['stft']
        shift = random.randint(-self.max_shift, self.max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        stft = np.pad(stft, ((0, 0), (a, b)), "constant")
        if a == 0:
            stft = stft[:,b:]
        else:
            stft = stft[:,0:-a]
        data['stft'] = stft
        return data

class FixSTFTDimension(object):
    def __call__(self, data):
        stft = data['stft']
        t_len = stft.shape[1]
        orig_t_len = data['stft_shape'][1]
        if t_len > orig_t_len:
            stft = stft[:,0:orig_t_len]
        elif t_len < orig_t_len:
            stft = np.pad(stft, ((0, 0), (0, orig_t_len-t_len)), "constant")
        data['stft'] = stft
        return data

class ToMelSpectrogramFromSTFT(object):
    def __init__(self, n_mels=32):
        self.n_mels = n_mels
    def __call__(self, data):
        stft = data['stft']
        sample_rate = data['sample_rate']
        n_fft = data['n_fft']
        mel_basis = librosa.filters.mel(sample_rate, n_fft, self.n_mels)
        s = np.dot(mel_basis, np.abs(stft)**2.0)
        data['mel_spectrogram'] = librosa.power_to_db(s, ref=np.max)
        return data

class DeleteSTFT(object):
    def __call__(self, data):
        del data['stft']
        return data

class ToTensor(object):
    def __init__(self, np_name, tensor_name, normalize=None):
        self.np_name = np_name
        self.tensor_name = tensor_name
        self.normalize = normalize
    def __call__(self, data):
        tensor = torch.FloatTensor(data[self.np_name])
        if self.normalize is not None:
            mean, std = self.normalize
            tensor -= mean
            tensor /= std
        data[self.tensor_name] = tensor
        return data