import random
import numpy as np
import argparse
import datetime
range_min=0
range_max=100
train_num=12000
test_num=1200
max_category=120

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
    train_data=[train_data_temp,train_data_temp]
    test_data=[test_data_temp,test_data_temp]
    np.savetxt("train.csv", np.array(train_data), delimiter=',')
    np.savetxt("test.csv", np.array(test_data), delimiter=',')