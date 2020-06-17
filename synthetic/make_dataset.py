import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, datasets
import random
import numpy as np

range_max=1
train_num=12000
test_num=1200

class train_dst(Dataset):
    
    def __init__(self, transform=None):
        self.transform = transform
        self.size = train_num

    def __len__(self):
        return self.size

    def __getitem__(self):
        train_data = np.loadtxt(open("train.csv","rb"), delimiter=",")
        train_label=np.zeros(train_num,dtype=int)
        for i in range(train_num):
            idx=int(i/(train_num/num_seg))
            train_label[i]=int(idx % 2)        
        sample = {'image': train_data, 'label': train_label}
        if self.transform:
            sample = self.transform(sample)

        return sample
class test_dst(Dataset):
    
    def __init__(self, transform=None):
        self.transform = transform
        self.size = test_num

    def __len__(self):
        return self.size

    def __getitem__(self):

        test_data = np.loadtxt(open("test.csv","rb"), delimiter=",")
        test_label=np.zeros(test_num,dtype=int)
        for i in range(test_num):
            idx=int(i/(test_num/num_seg))
            test_label[i]=int(idx % 2)
        sample = {'image': test_data, 'label': test_label}
        if self.transform:
            sample = self.transform(sample)
        return sample