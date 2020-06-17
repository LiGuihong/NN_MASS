import torch 
import torch.nn as nn 
import torchvision.datasets as dsets 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import random
import numpy as np


from torchvision.models import vgg16
from collections import namedtuple

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained = True).features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,15,22}:
                results.append(x)
'''
Hyper-parameter definition
'''
#define the scaling factor for the network's width
arch_scale=0.5

#define the depth of the network
net_depth=9
num_all_path=int(((net_depth-2)*(net_depth-1))/2)
layer_density=np.zeros(net_depth-2)
short_cut_num_record=np.ones(net_depth-2) #the number of layer's connection
short_cut_record=np.zeros([net_depth-2,net_depth])# the index between some layer with former layer
#the number of the decision district, start with 2
num_seg=2 #
range_min=0
range_max=1
train_num=12000
test_num=1200
max_category=120

#define the number of neuron in each FC layer

net_arch=[32,32,32,32,32,32,32,32,32]
net_name=['fc1','fc2','fc3','fc4','fc5','fc6','fc7','fc8','fc9']

#short_path is generated automatically by generate_shortcut

# [1,3]  [1,4][2,4]  [1,5][2,5][3,5] 
# [1,6][2,6][3,6][4,6] 
# [1,7][2,7][3,7][4,7][5,7]
all_short_cut=[] # 
short_cut=[]
shortcut_num=20 #tc
tc_list=[10,20,30,20,10,30,40,60,60]
shortcut_tc=np.zeros(num_all_path,dtype=int)



# data generation
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
    #print(len(train_data[1]))
    np.savetxt("train.csv", np.array(train_data), delimiter=',')
    np.savetxt("test.csv", np.array(test_data), delimiter=',')
#data_generation()


def make_dataset(train_file='train.csv',test_file='test.csv'):
    train_data = np.loadtxt(open("train.csv","rb"), delimiter=",")
    test_data = np.loadtxt(open("test.csv","rb"), delimiter=",")
    train_label=np.zeros(train_num,dtype=int)
    test_label=np.zeros(test_num,dtype=int)
    for i in range(train_num):
        idx=int(i/(train_num/num_seg))
        train_label[i]=int(i % 2)

    for i in range(test_num):
        idx=int(i/(test_num/num_seg))
        test_label[i]=int(i % 2)
    
    return train_data,test_data,train_label,test_label

# [1,3] [1,4][2,4] [1,5][2,5][3,5] 
# [1,6][2,6][3,6][4,6] 
# [1,7][2,7][3,7][4,7][5,7]

def shortcut_nn_mass():
    if net_depth<3:
        return None
    else:
        for i in range(net_depth-2):
            for j in range(i):
                short_path=[j+1,net_depth]
                all_short_cut.append(short_path);
    all_path_num=np.zeros(net_depth-2)
    for i in range(net_depth-2):
        for j in range(i):
            all_path_num[i]=all_path_num[i]+net_arch[i]*(net_arch[j+2])


    shortcut_num=np.min(num_all_path,shortcut_num)

    short_cut=random.choice(all_short_cut,shortcut_num)
    for i in range(shortcut_num):
        shortcut_tc[i]=int(random.uniform(0.499*net_arch[short_cut[i,0]],1.001*net_arch[short_cut[i,0]]))
        short_cut_record[short_cut[i,1]]=short_cut[i,0]
        short_cut_num_record[short_cut[i,1]]=+1

        #layer_density[short_cut[i,1]]=layer_density[short_cut[i,1]]+shortcut_tc[i]/all_path_num[short_cut[i,1]]
    density=(np.sum(shortcut_tc))/np.sum(all_path_num)
    nn_mass=density*net_arch[0]*net_depth
    return nn_mass



            


class SimpleNet(nn.Module):
    def __init__(self, args):
        super(SimpleNet, self).__init__()
        self.add_module(net_name[0],nn.Linear(2, net_arch[0]))# 2 is the dimensions of the data
        for i in range(net_depth-2):
            # nn.Linear(input_size, num_classes)
            self.add_module(net_name[i+1],nn.Linear(net_arch[i], net_arch[i+1]))
        self.add_module(net_name[net_depth-1],2)  #2 is the number of the class
        
        self.link_dict={}
        self.Q1_3=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q1_4=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q2_4=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q1_5=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q2_5=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q3_5=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q1_6=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q2_6=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q3_6=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q4_6=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q1_7=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q2_7=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q3_7=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q4_7=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q5_7=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q1_8=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q2_8=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q3_8=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q4_8=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q5_8=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q6_8=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q1_9=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q2_9=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q3_9=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q4_9=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q5_9=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q6_9=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        self.Q7_9=nn.Parameter(torch.zeros(32,32),requires_grad=False)
        for i in range(shortcut_num):

        
'''
        for i in range(shortcut_num):
            shortcut_name='self.Q'+str(short_cut[i,0])+'_'+str(short_cut[i,1])
            post_fix='nn.Parameter(torch.empty('+str(shortcut_tc)+','+str(net_arch[2])+').normal_(mean=0, std=0.01),requires_grad=False)'
            add_link=shortcut_name+post_fix
            exec(add_link)

            idx_tmp=random.choice(np.arange(net_arch[short_cut[i,0]],dtype=int),net_arch[short_cut[i,0]]-shortcut_tc[i])
            short_cut_idx=np.ones(net_arch[short_cut[i,0]])
            for nums in idx_tmp:
                short_cut_idx[nums]=0
            #self.l1_3=random.choice(np.arange(net_arch[short_cut[i,0]],dtype=int),shortcut_tc[i])
            shortcut_name='self.idx'+str(short_cut[i,0])+'_'+str(short_cut[i,1])
            #post_fix='=random.choice(np.arange('+str(net_arch[short_cut[i,0]]) \
            #    +',dtype=int),'+str(shortcut_tc[i]) +')'
            post_fix='=Tensor(short_cut_idx)'
            short_cut_idx=shortcut_name+post_fix
            exec(short_cut_idx)
'''


    def forward(self, x):
        out1=F.relu(self.fc1(x))
        
        out2=F.relu(self.fc2(out1))
        for i in range(net_depth-2):
            #out  3 = self.fc 3 (out 2 )
            fc_code='out'+str(i+3)+'self.fc'+str(i+3)+'(out' +str(i+2)+')'
            exec(fc_code)
            for j in range(short_cut_num_record[i]):
                idx=short_cut_num_record[j]

                #out3=out3+self.l idx_3*out idx
                res_code='out'+str(i+3)+'=out'+str(i+3)+'+'+'self.l'+str(idx)+'_'+str(i+3)+'*out'+str(idx)
                exec(res_code)                
        out=[]
        output='out=out'+str(net_depth+1)
        exec(output)
        return out




'''
model = SimpleNet()

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

train_data,test_data,train_label,test_label=make_dataset()
# Training the Model
def train_model():
    for epoch in range(num_epochs):
        data = train_data
        labels = train_label

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        # (1)
        loss.backward()
        # (2)
        optimizer.step()
        # (3)

        if (i + 1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1,
                        len(train_data) // batch_size, loss.data.item()))

# Test the Model
correct = 0
total = 0
#for images, labels in test_loader:
def test_model():
    images = test_data
    labels = test_label
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: % d %%' % (100 * correct / total))

    
#'''


    

