from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from torch.autograd import Variable
train_num=12000
num_seg=2
class my_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_file, label_file, train_num=12000,num_seg=2,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.label = np.loadtxt(open(label_file,"rb"), delimiter=",",dtype=float).T
        self.data  = np.loadtxt(open(data_file,"rb"), delimiter=",",dtype=float).T
        self.transform = transform
        self.train_num=train_num
        self.num_seg=num_seg
    def __len__(self):
        return self.train_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx,:]
        landmarks=int(idx/(int((self.train_num/self.num_seg))))%2
        #landmarks = self.label[idx]
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        return sample

train_dataset = my_dataset(data_file='train.csv',label_file='tr.csv',train_num=train_num,num_seg=num_seg)


train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = 60,shuffle = True)
print(train_loader)
for x in enumerate(train_loader):

    images = x[0]
    labels = x[1]
    print(type(x[0]))
    print(type(x[1]))


