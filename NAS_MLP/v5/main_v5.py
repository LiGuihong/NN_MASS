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

#-------------------------------------
file_logs=open('logs5_v1.txt','a+')
train_logs=open('train5_v1.txt','a+')
# argument parser
parser = argparse.ArgumentParser(description='ML_CODESIGN Lab1 - MNIST example')
parser.add_argument('--batch_size', type=int, default=100, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--depth', type=int, default=8, help='Learning rate')
parser.add_argument('--width', type=int, default=8, help='Learning rate')
parser.add_argument('--num_seg', type=int, default=6, help='Learning rate')
parser.add_argument('--shortcut_num', type=int, default=20, help='Learning rate')
args = parser.parse_args()
shortcut_num=args.shortcut_num
learning_rate=args.lr
num_epochs=args.epochs
batch_size=args.batch_size
net_width=args.width
num_seg=args.num_seg
'''
if args.depth>5:
    num_epochs=1000
if args.depth>8:
    num_epochs=1500
if args.depth>10:
    num_epochs=2000
'''

'''
Hyper-parameter definition
'''
#define the scaling factor for the network's width
arch_scale=0.5

#define the depth of the network
net_depth=args.depth
num_all_path=int(((net_depth-2)*(net_depth-1))/2)
layer_density=np.zeros(net_depth-2)
short_cut_num_record=np.zeros(net_depth,dtype=int) #the number of layer's connection
short_cut_record=np.zeros([net_depth,net_depth],dtype=int)# the index between some layer with former layer
#the number of the decision district, start with 2
 #
range_min=0
range_max=100
train_num=60000
test_num=12000
max_category=120

#define the number of neuron in each FC layer

net_arch=[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
for i in range(len(net_arch)):
    net_arch[i]=net_width

net_name=['fc0','fc1','fc2','fc3','fc4','fc5','fc6','fc7','fc8','fc9','fc10','fc11','fc12','fc13','fc14','fc15','fc16','fc17','fc18','fc19']

#short_path is generated automatically by generate_shortcut

# [1,3]  [1,4][2,4]  [1,5][2,5][3,5] 
# [1,6][2,6][3,6][4,6] 
# [1,7][2,7][3,7][4,7][5,7]
all_short_cut=[] # 

short_cut=None
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
    np.savetxt("train5_data.csv", np.array(train_data), delimiter=',')
    np.savetxt("test5_data.csv", np.array(test_data), delimiter=',')
#data_generation()





#************************************************************
#************************************************************
if net_depth<3:
    print('dpeth<3')
    
else:
    for i in range(net_depth-2):
        for j in range(i+1):
            short_path=[j,i+2]
            all_short_cut.append(short_path);
all_path_num=np.zeros(net_depth-2)
for i in range(net_depth-2):
    for j in range(i+1):
        all_path_num[i]=all_path_num[i]+(net_arch[j])


shortcut_num=min(num_all_path,shortcut_num)
short_cut_idc=random.sample(list(np.arange(num_all_path)),shortcut_num)
tmp=np.array(all_short_cut)

short_cut=tmp[short_cut_idc,:]

for i in range(shortcut_num):
    shortcut_tc[i]=int(random.uniform(0.499*net_arch[short_cut[i,0]],1.001*net_arch[short_cut[i,0]]))
    #print(str(short_cut[i,0])+str(short_cut[i,1])+str(short_cut_num_record[short_cut[i,1]]))

    short_cut_record[short_cut[i,1],short_cut_num_record[short_cut[i,1]]]=short_cut[i,0]
    short_cut_num_record[short_cut[i,1]]=short_cut_num_record[short_cut[i,1]]+1
    #print(short_cut_record[:,0:2])
    #layer_density[short_cut[i,1]]=layer_density[short_cut[i,1]]+shortcut_tc[i]/all_path_num[short_cut[i,1]]
density=(np.sum(shortcut_tc))/np.sum(all_path_num)
nn_mass=density*net_arch[0]*net_depth



net_arch[net_depth-1]=2
#************************************************************
#************************************************************
            


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        layer_list=[]
        layer_list.append(nn.Linear(2, net_arch[0]))
        for i in range(net_depth-2):
            layer_list.append(nn.Linear(net_arch[i], net_arch[i+1]))  
        layer_list.append(nn.Linear(net_arch[net_depth-2], 2))     
        self.features = nn.ModuleList(layer_list).eval() 
#        self.layer_dict={}
#        self.layer_dict[net_name[0]]=nn.Linear(2, net_arch[0])        
#        for i in range(net_depth-2):
#            self.layer_dict[net_name[i+1]]=nn.Linear(net_arch[i], net_arch[i+1])
#        self.layer_dict[net_name[net_depth-1]]=nn.Linear(net_arch[net_depth-1], 2)

        
        self.link_dict={}
        for i in range(shortcut_num):

            link_name='l'+str(short_cut[i,0])+'_'+str(short_cut[i,1])
            self.link_dict[link_name]=self.add_link(i)


    def add_link(self,idx=0):
        #link_params=nn.Parameter(torch.zeros([net_arch[short_cut[idx,0]],net_arch[short_cut[idx,1]]]),requires_grad=False)
        link_params=torch.zeros([net_arch[short_cut[idx,0]],net_arch[short_cut[idx,1]]])
        for i in range(net_arch[short_cut[idx,1]]):
            tmp=list((np.arange(net_arch[short_cut[idx,0]])))
            link_idx=random.sample(tmp,shortcut_tc[idx])
            
            link_params[link_idx,i]=1
        link_params=link_params#.cuda()
        return link_params


    def forward(self, x):

        out0=F.relu(self.features[0](x))        
        out1=F.relu(self.features[1](out0))
        out_dict={}
        out_dict['0']=out0
        out_dict['1']=out1
        for layer_idx in range(net_depth-2):
   
            out_tmp=self.features[layer_idx+2](out_dict[str(layer_idx+1)])
            out_dict[str(layer_idx+2)]=out_tmp
            for k in range(short_cut_num_record[layer_idx+2]):
                link_name='l'+str(short_cut_record[layer_idx+2,0])+'_'+str(layer_idx+2)
                link_temp=self.link_dict[link_name]
                #link_temp=torch.unsqueeze(link_temp,dim=0)
                #link=torch.cat([link_temp] * batch_size, dim=0)
                tmp=torch.matmul(out_dict[str(short_cut_record[layer_idx,0])],link_temp)
                out_tmp=out_tmp+tmp
            out_dict[str(layer_idx+2)]=F.relu(out_tmp)

        return out_dict[str(net_depth-1)]

def make_dataset(train_file='train.csv',test_file='test.csv'):
    train_data = np.loadtxt(open("train5_data.csv","rb"), delimiter=",",dtype=float)
    test_data = np.loadtxt(open("test5_data.csv","rb"), delimiter=",",dtype=float)
    train_label=np.zeros(train_num,dtype=int)
    test_label=np.zeros(test_num,dtype=int)
    for i in range(train_num):
        idx=int(i/(train_num/num_seg))
        train_label[i]=int(idx % 2)
    np.savetxt("tr5.csv", np.array(train_label), delimiter=',')
    for i in range(test_num):
        idx=int(i/(test_num/num_seg))
        test_label[i]=int(idx % 2)
    np.savetxt("te5.csv", np.array(test_label), delimiter=',')
    return train_data.T,test_data.T,train_label.T,test_label.T

[train_data_raw,test_data,train_label_raw,test_label]=make_dataset()
raw_label=np.zeros([train_num,2])
for i in range(train_num):
    raw_label[i,train_label_raw[i]]=1
train_data_raw=torch.tensor(train_data_raw,dtype=torch.float)#.cuda()
test_data=torch.tensor(test_data,dtype=torch.float)#.cuda()
train_label_raw=Variable(torch.from_numpy(train_label_raw))#.cuda()
test_label=Variable(torch.from_numpy(test_label))#.cuda()


#print('*************************************************************',file=file_logs)
#print('*************************************************************',file=file_logs)

#print('*************************************************************',file=train_logs)
#print('*************************************************************',file=train_logs)
print('    ,'+str(nn_mass) +', '+str(net_width)+', '+str(net_depth)+', '+str(density),file=train_logs) 

for iter_times in range(7):
    model = SimpleNet()#.cuda()

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        perm_idx=torch.randperm(train_num)
        train_data=train_data_raw[perm_idx]
        train_label=train_label_raw[perm_idx]
        steps=int(train_num/batch_size)
        train_data=train_data.view([steps,batch_size,-1])
        train_label=train_label.view([steps,batch_size])
        correct = 0
        total = 0
        

        for i in range(steps):
            image = Variable(train_data[i])
            label = Variable(train_label[i])
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            # (1)
            loss.backward()
            # (2)
            optimizer.step()
            # (3)

            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum()
        print(str(100 * float(correct) / total),file=train_logs)
        #print('Train Accuracy after epoch '+str(epoch+1) +' ,' +str(100 * float(correct) / total))
    #print('------******------******------******------******------******------******------',file=train_logs)
    # Test the Model
    correct = 0
    total = 0

    test_data=test_data.view([-1,batch_size,2])
    test_label=test_label.view([-1,batch_size])
    steps= int(test_num/batch_size)
    for i in range(steps):
        image = Variable(test_data[i])
        label = Variable(test_label[i])
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum()

    #print('------******------******------******------******------******------******------',file=file_logs)
    print(str(nn_mass) +', '+str((100 * float(correct) / total))+', '+str(net_width)
        +', '+str(net_depth)+', '+str(density)+', '+str(num_seg),file=file_logs) 
    print('Accuracy of the model on the 10000 test images: % f %%' % (100 * float(correct) / total))
#'''