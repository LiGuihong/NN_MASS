import random
import numpy as np
import argparse
import datetime
range_min=0
range_max=100
train_num=12000
test_num=1200
max_category=120
data_dim=32
def data_generation():
    train_data_temp=[]
    train_label=np.zeros(train_num,dtype=int)
    test_data_temp=[]
    test_label=np.zeros(test_num,dtype=int)
    steps=(range_max-range_min)/(max_category)
    
    for i in range(max_category):
        counter = 1 
        while(counter<=(train_num/max_category)): 
            temp_rand=random.uniform(range_min+i*steps,range_min+(i+1)*steps)
            if(temp_rand not in train_data_temp):
                train_data_temp.append(temp_rand); 
                counter+=1
    for i in range(max_category):
        counter = 1 
        while(counter<=(test_num/max_category)): 
            temp_rand=random.uniform(range_min+i*steps,range_min+(i+1)*steps)
            if(temp_rand not in test_data_temp):
                if(temp_rand not in train_data_temp):
                    test_data_temp.append(temp_rand); 
                    counter+=1
    train_data=np.zeros([train_num,data_dim])
    test_data=np.zeros([test_num,data_dim])
    for i in range(data_dim):
        train_data[:,i]=train_data_temp
        test_data[:,i]=test_data_temp
    np.savetxt("data/train_32.csv", np.array(train_data), delimiter=',')
    np.savetxt("data/test_32.csv", np.array(test_data), delimiter=',')

#data_generation()


range_min=0
range_max=100
train_num=12000
test_num=1200
max_category=120
data_dim=32
def data_generation_random():
    train_data_rand=[]
    train_data_temp=[]
    train_label=np.zeros(train_num,dtype=int)
    test_data_temp=[]
    test_data_rand=[]
    test_label=np.zeros(test_num,dtype=int)
    steps=(range_max-range_min)/(max_category)
    
    for i in range(max_category):
        counter = 1 
        while(counter<=(train_num/max_category)): 
            temp_rand=random.uniform(range_min+i*steps,range_min+(i+1)*steps)
            viat_x=np.random.normal(0,0.5,1)
            viat_y=np.random.normal(0,0.5,1)
            temp_rand_x=temp_rand+viat_x
            temp_rand_y=temp_rand+viat_y
            tmp=[temp_rand_x,temp_rand_y]
            if(tmp not in train_data_rand):
                train_data_rand.append(tmp); 
                counter+=1
            if([temp_rand,temp_rand] not in train_data_temp):
                train_data_temp.append([temp_rand,temp_rand]); 
    for i in range(max_category):
        counter = 1 
        while(counter<=(test_num/max_category)): 
            temp_rand=random.uniform(range_min+i*steps,range_min+(i+1)*steps)
            viat_x=np.random.normal(0,0.5,1)
            viat_y=np.random.normal(0,0.5,1)
            temp_rand_x=temp_rand+viat_x
            temp_rand_y=temp_rand+viat_y
            tmp=[temp_rand_x,temp_rand_y]
            if(tmp not in test_data_rand):
                if(tmp not in train_data_rand):
                    test_data_rand.append(tmp); 
                    counter+=1
            if([temp_rand,temp_rand]  not in test_data_temp):
                if([temp_rand,temp_rand]  not in train_data_temp):
                    test_data_temp.append([temp_rand,temp_rand] ); 
    train_data=np.squeeze(np.array(train_data_temp))
    #print(train_data.size)
    test_data=np.squeeze(np.array(test_data_temp))
 
    np.savetxt("data/train.csv", np.array(train_data), delimiter=',')
    np.savetxt("data/test.csv", np.array(test_data), delimiter=',')

    train_data=np.squeeze(np.array(train_data_rand))
    test_data=np.squeeze(np.array(test_data_rand))
    np.savetxt("data/train_random.csv", np.array(train_data), delimiter=',')
    np.savetxt("data/test_random.csv", np.array(test_data), delimiter=',')  

data_generation_random()


'''
range_min=0
range_max=100
train_num=65536
test_num=4096
max_category=256

def data_generation_65536():
    train_data_temp=[]
    train_label=np.zeros(train_num,dtype=int)
    test_data_temp=[]
    test_label=np.zeros(test_num,dtype=int)
    steps=(range_max-range_min)/(max_category)
    
    for i in range(max_category):
        counter = 1 
        while(counter<=(train_num/max_category)): 
            temp_rand=random.uniform(range_min+i*steps,range_min+(i+1)*steps)
            if(temp_rand not in train_data_temp):
                train_data_temp.append(temp_rand); 
                counter+=1
    for i in range(max_category):
        counter = 1 
        while(counter<=(test_num/max_category)): 
            temp_rand=random.uniform(range_min+i*steps,range_min+(i+1)*steps)
            if(temp_rand not in test_data_temp):
                if(temp_rand not in train_data_temp):
                    test_data_temp.append(temp_rand); 
                    counter+=1
    train_data=[train_data_temp,train_data_temp]
    test_data=[test_data_temp,test_data_temp]
    np.savetxt("train_65536.csv", np.array(train_data), delimiter=',')
    np.savetxt("test_65536.csv", np.array(test_data), delimiter=',')
data_generation_65536()
'''

