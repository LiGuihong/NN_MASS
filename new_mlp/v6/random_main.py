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
import datetime
import dataset as my_dataset
#-------------------------------------

file_logs=open('random_log.txt','a+')
train_logs=open('train_random.txt','a+')
feat_logs=open('feat_random.txt','a+')
# argument parser
parser = argparse.ArgumentParser(description='ML_CODESIGN Lab1 - MNIST example')
parser.add_argument('--batch_size', type=int, default=100, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--depth', type=int, default=4, help='Learning rate')
parser.add_argument('--width', type=int, default=8, help='Learning rate')
parser.add_argument('--num_seg', type=int, default=2, help='Learning rate')
parser.add_argument('--shortcut_num', type=int, default=20, help='Learning rate')
parser.add_argument('--tc', type=int, default=20, help='the number of tc')
args = parser.parse_args()
net_depth=args.depth
tc=args.tc
net_width=args.width


net_depth=int(random.uniform(6,10))
num_all_path=int(((net_depth-2)*(net_depth-1))/2)
tc=int(random.uniform(num_all_path/2,num_all_path))



learning_rate=args.lr
num_epochs=args.epochs
batch_size=args.batch_size
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

layer_density=np.zeros(net_depth-2)
#the number of the decision district, start with 2
range_min=my_dataset.range_min
range_max=my_dataset.range_max
train_num=my_dataset.train_num
test_num=my_dataset.test_num
max_category=my_dataset.max_category

#define the number of neuron in each FC layer

net_arch=[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
for i in range(len(net_arch)):
    net_arch[i]=net_width

net_name=['fc0','fc1','fc2','fc3','fc4','fc5','fc6','fc7','fc8','fc9','fc10','fc11','fc12','fc13','fc14','fc15','fc16','fc17','fc18','fc19']
all_short_cut=[] 

if net_depth<3:
    print('depth<3, depth must>=3. program will exit due to the invalid depth definition')
    exit(0)
all_path_num=np.zeros(net_depth)
layer_tc=np.zeros(net_depth)
for i in range(net_depth-2):
    for j in range(i+1):
        all_path_num[i+2]=all_path_num[i+2]+(net_arch[j])
    layer_tc[i+2]=min(tc,all_path_num[i+2])

layer_tc=np.array(layer_tc,dtype=int)
all_path_num=np.array(all_path_num,dtype=int)
density=(np.sum(layer_tc))/(np.sum(all_path_num))
nn_mass=density*net_arch[0]*net_depth
#print(net_depth)
#print(all_path_num)
#print(nn_mass)
#exit(0)
class SimpleNet_order_no_batch(torch.nn.Module):
    def __init__(self):
        super(SimpleNet_order_no_batch, self).__init__()
        batchnorm_list=[]
        layer_list=[]
        layer_list.append(nn.Linear(2, net_arch[0]+layer_tc[0]))
        batchnorm_list.append(nn.BatchNorm1d(net_arch[0]+layer_tc[0]))
        for i in range(net_depth-2):
            layer_list.append(nn.Linear(net_arch[i+1]+layer_tc[i+1], net_arch[i+1]))
            batchnorm_list.append(nn.BatchNorm1d(net_arch[i+1]+layer_tc[i+1]))  
        layer_list.append(nn.Linear(net_arch[net_depth-1]+layer_tc[net_depth-1], 2))
        self.features = nn.ModuleList(layer_list).eval() 
        self.batchnorm = nn.ModuleList(batchnorm_list).eval()
        self.link_dict=[]
        for i in range(net_depth):
            self.link_dict.append(self.add_link(i))
        self.params,self.flops=self.param_num()
        
    def param_num(self):
        num_param=0
        flops=0
        for layer in self.features:
            num_param=num_param+(layer.in_features)*(layer.out_features)+layer.out_features
            flops=flops+2*(layer.in_features)*(layer.out_features)+layer.out_features
        return num_param,flops

    def add_link(self,idx=0):
        tmp=list((np.arange(all_path_num[idx])))
        link_idx=random.sample(tmp,layer_tc[idx])
        link_params=torch.tensor(link_idx)
        return link_params
    def forward(self, x):
        out0=F.relu(self.features[0](x))        
        out1=self.features[1](out0)
        out_dict=[]
        out_dict.append(out0)
        out_dict.append(out1)
        feat_dict=[]
        feat_dict.append(out0)
        feat_dict.append(torch.cat((out1,out0),1))
        for layer_idx in range(net_depth-2):
            in_features=feat_dict[layer_idx]
            if layer_tc[layer_idx+2]>0:
                in_tmp=torch.cat((out_dict[layer_idx+1],in_features[:,self.link_dict[layer_idx+2]]),1)
                if layer_idx<net_depth-3:
                    out_tmp=self.features[layer_idx+2](F.relu(in_tmp))
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp=self.features[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)           
            else:
                in_tmp=out_dict[layer_idx+1]
                if layer_idx<net_depth-3:
                    out_tmp=self.features[layer_idx+2](F.relu(in_tmp))
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp=self.features[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)     
        return out_dict[net_depth-1]


def make_dataset(train_file='train.csv',test_file='test.csv'):
    train_data = np.loadtxt(open(train_file,"rb"), delimiter=",",dtype=float)
    test_data = np.loadtxt(open(test_file,"rb"), delimiter=",",dtype=float)
    train_label=np.zeros(train_num,dtype=int)
    test_label=np.zeros(test_num,dtype=int)
    for i in range(train_num):
        idx=int(i/(train_num/num_seg))
        train_label[i]=int(idx % 2)
    np.savetxt("tr_random.csv", np.array(train_label), delimiter=',')
    for i in range(test_num):
        idx=int(i/(test_num/num_seg))
        test_label[i]=int(idx % 2)
    np.savetxt("te_random.csv", np.array(test_label), delimiter=',')
    return train_data.T,test_data.T,train_label.T,test_label.T

[train_data_raw,test_data,train_label_raw,test_label]=make_dataset()
raw_label=np.zeros([train_num,2])
for i in range(train_num):
    raw_label[i,train_label_raw[i]]=1
train_data_raw=torch.tensor(train_data_raw,dtype=torch.float)
test_data=torch.tensor(test_data,dtype=torch.float)
train_label_raw=Variable(torch.from_numpy(train_label_raw))
test_label=Variable(torch.from_numpy(test_label))


#print('*************************************************************',file=file_logs)
#print('*************************************************************',file=file_logs)

#print('*************************************************************',file=train_logs)
#print('*************************************************************',file=train_logs)
print('    ,'+str(nn_mass) +', '+str(net_width)+', '+str(net_depth)+', '+str(density),file=train_logs) 

def find_bound(predicted):
    bound=[]
    bound_num=0
    for i in range(test_num-1):
        if predicted[i]!=predicted[i+1]:
            bound.append(i)
            bound_num+=1
    step=int(min(bound_num,num_seg-1))

    return bound_num,bound

for iter_times in range(5):
    model = SimpleNet_order_no_batch()
    print(model.features,file=feat_logs)
    #print(model.link_dict)
    #print(model.li)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    print(datetime.datetime.now())
    max_acc=-1000
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
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum()
        if max_acc<float(correct) / total:
            max_acc=float(correct) / total
        print(str(100 * float(correct) / total),file=train_logs)
        #print('Train Accuracy after epoch '+str(epoch+1) +' ,' +str(100 * float(correct) / total))
    #print('------******------******------******------******------******------******------',file=train_logs)
    # Test the Model
    correct = 0
    total = 0
    predicted_label=torch.zeros(test_num)
    test_data=test_data.view([-1,batch_size,2])
    test_label=test_label.view([-1,batch_size])
    steps= int(test_num/batch_size)
    for i in range(steps):
        image = Variable(test_data[i])
        label = Variable(test_label[i])
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label[i*batch_size:(i+1)*batch_size]=predicted
        total += label.size(0)
        correct += (predicted == label).sum()
    bound_num,bound=find_bound(predicted_label.cpu())
    #print('------******------******------******------******------******------******------',file=file_logs)
    print(str(nn_mass) +', '+str((100 * float(correct) / total))+','+str(100*max_acc)+', '
        +str(model.params)+', '+str(model.flops)+', '+str(net_width)+', '+str(net_depth)+', '
        +str(tc)+', '+str(num_seg)+', '+str(bound_num)+', '+str(bound),file=file_logs)  
    print('Accuracy of the model on the 10000 test images: % f %%' % (100 * float(correct) / total))
    #print(datetime.datetime.now())