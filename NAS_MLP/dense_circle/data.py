import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, datasets
import random
import numpy as np
import argparse
import math


range_min=0
range_max=100
train_num=12000
test_num=1200
max_category=120

def data_generation():
    train_data_temp=[]
    train_data_temp_1=[]
    train_label=np.zeros(train_num,dtype=int)
    test_data_temp=[]
    test_label=np.zeros(test_num,dtype=int)
    steps=(range_max-range_min)/(max_category)
    
    for i in range(max_category):
        counter = 1 
        mu=(range_min+i*steps+range_min+(i+1)*steps)/2
        sig=steps/2
        while(counter<=(train_num/max_category)): 
            len_temp=(np.random.normal(mu, sig, 1))
            #temp_rand=random.uniform(range_min+i*steps,range_min+(i+1)*steps)
            if len_temp < (i+1)*steps and len_temp>(i)*steps:
                theta=random.uniform(0,2*3.14159265357)
                temp_rand_1=len_temp*math.cos(theta)
                temp_rand_2=len_temp*math.sin(theta)
                temp_rand=[temp_rand_1,temp_rand_2]
                if(temp_rand not in train_data_temp):
                    train_data_temp.append(temp_rand); 
                    counter+=1
                    #print(counter)
                    

    for i in range(max_category):
        counter = 1 
        mu=(range_min+i*steps+range_min+(i+1)*steps)/2
        sig=steps/2
        while(counter<=(test_num/max_category)): 
            len_temp=(np.random.normal(mu, sig, 1))
            if len_temp < (i+1)*steps and len_temp>(i)*steps:
                theta=random.uniform(0,2*3.14159265357)
                temp_rand_1=len_temp*math.cos(theta)
                temp_rand_2=len_temp*math.sin(theta)
                temp_rand=[temp_rand_1,temp_rand_2]
                if(temp_rand not in test_data_temp):
                    if(temp_rand not in train_data_temp):
                        test_data_temp.append(temp_rand); 
                        counter+=1
                        #print(counter)
    train_data=np.squeeze(train_data_temp)
    test_data=np.squeeze(test_data_temp)
    np.savetxt("train.csv", np.array(train_data), delimiter=',')
    np.savetxt("test.csv", np.array(test_data), delimiter=',')

data_generation()


