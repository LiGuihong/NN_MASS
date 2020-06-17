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

#-------------------------------------
file_logs=open('logs/log_lin_v0.txt','a+')
train_logs=open('logs/train_lin_v0.txt','a+')
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


net_depth=args.depth
net_width=args.width

learning_rate=args.lr
num_epochs=args.epochs
batch_size=args.batch_size
num_seg=args.num_seg
tc=args.tc


layer_density=np.zeros(net_depth-2)
#define the number of neuron in each FC layer

net_arch=[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
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
        self.depth=net_depth
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

    def singular(self, x):
        x.requires_grad = True
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
        y=out_dict[net_depth-1]
        sigma_all=[]
        for i in range(101):
            out_00=torch.autograd.grad(y[i,0], x,  retain_graph=True)
            out_11=torch.autograd.grad(y[i,1], x,  retain_graph=True)
            
            out_0=out_00[0][i]
            out_1=out_11[0][i]
            #print(out_0)
            #print(out_1)
            out_0=out_0.view([1,2])
            out_1=out_1.view([1,2])
            #print(out_0)
            #print(out_1)
            out_all=torch.cat((out_0,out_1),axis=1)
            #print(out_all)
            _,sigma,_=torch.svd(out_all)
            sigma_all.append(sigma) 
        sigma_all=np.array(sigma_all)
        #print(np.mean(sigma_all))
        return np.mean(sigma_all)


mean_file='sigma/sigma_init_'+str(net_width)+'_'+str(net_depth)+'_'+str(tc)+'_'+str(nn_mass)+'.csv'
x_raw=np.arange(10001,dtype=float)/10000
x=np.array([x_raw,x_raw])
x=x.transpose()
sig_value=[]
for i in range(10):
    model = SimpleNet_order_no_batch()
    x=torch.tensor(x,dtype=torch.float)
    sig_value.append(model.singular(x))

file_logs=open('logs/sigma.txt','a+')
print(np.mean(np.array(sig_value)),file=file_logs)
print(nn_mass,file=file_logs)
np.savetxt(mean_file, np.array(sig_value), delimiter=',')