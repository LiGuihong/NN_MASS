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
train_num=24000
test_num=2400
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


def square_data(vert_min=0.0,vert_max=30.0,hort_min=0,hort_max=30,vert_seg=4,hort_seg=5):
    train_data_temp=[]
    train_data_temp_1=[]
    train_label=np.zeros(train_num,dtype=int)
    test_data_temp=[]
    test_label=np.zeros(test_num,dtype=int)
    max_category=vert_seg * hort_seg
    steps=(range_max-range_min)/(max_category)
    hort_step=(hort_max-hort_min)/hort_seg
    vert_step=(vert_max-vert_min)/vert_seg
    for i in range(vert_seg):
        for j in range(hort_seg):
            counter = 1 
            hort_mu=(hort_min+i*hort_step+hort_min+(i+1)*hort_step)/2
            hort_sig=hort_step*1.5

            vert_mu=(vert_min+i*vert_step+vert_min+(i+1)*vert_step)/2
            vert_sig=vert_step*1.5
            while(counter<=(train_num/max_category)): 
                hort_temp=(np.random.normal(hort_mu, hort_sig, 1))
                if hort_temp < (i+1)*hort_step and hort_temp>(i)*hort_step:
                    vert_temp=(np.random.normal(vert_mu, vert_sig, 1))
                    if vert_temp < (i+1)*vert_step and vert_temp>(i)*vert_step:

                        temp_rand=[hort_temp,vert_temp]
                        if(temp_rand not in train_data_temp):
                            train_data_temp.append(temp_rand); 
                            train_label[counter-1]=int((i+j)%2)
                            counter+=1
                            #print(counter)                    
    for i in range(vert_seg):
        for j in range(hort_seg):
            counter = 1 
            hort_mu=(hort_min+i*hort_step+hort_min+(i+1)*hort_step)/2
            hort_sig=hort_step*1.5

            vert_mu=(vert_min+i*vert_step+vert_min+(i+1)*vert_step)/2
            vert_sig=vert_step*1.5
            while(counter<=(test_num/max_category)): 
                hort_temp=(np.random.normal(hort_mu, hort_sig, 1))
                if hort_temp < (i+1)*hort_step and hort_temp>(i)*hort_step:
                    vert_temp=(np.random.normal(vert_mu, vert_sig, 1))
                    if vert_temp < (i+1)*vert_step and vert_temp>(i)*vert_step:

                        temp_rand=[hort_temp,vert_temp]
                        if(((temp_rand not in train_data_temp) and (temp_rand not in test_data_temp))):
                            test_data_temp.append(temp_rand); 
                            test_label[counter-1]=int((i+j)%2)
                            counter+=1
                            
                            #print(counter)
    train_data=np.squeeze(train_data_temp)
    test_data=np.squeeze(test_data_temp)
    np.savetxt("train.csv", np.array(train_data), delimiter=',')
    np.savetxt("test.csv", np.array(test_data), delimiter=',')    

    np.savetxt("train_label.csv", train_label, delimiter=',')
    np.savetxt("test_label.csv", test_label, delimiter=',') 
x=np.zeros(2)
x[0]=0.0
x[1]=1.0
x=np.array(x,dtype=int)
print(x)
#square_data()


