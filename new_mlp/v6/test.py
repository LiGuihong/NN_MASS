import torch
import torch.nn as nn 
from thop import profile
import numpy as np
import random
import torch.nn.functional as F
net_depth=6
tc=10
net_arch=[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]

all_path_num=np.zeros(net_depth)
layer_tc=np.zeros(net_depth)
for i in range(net_depth-2):
    for j in range(i+1):
        all_path_num[i+2]=all_path_num[i+2]+(net_arch[j])
    layer_tc[i+2]=min(tc,all_path_num[i+2])

layer_tc=np.array(layer_tc,dtype=int)
all_path_num=np.array(all_path_num,dtype=int)
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

model = SimpleNet_order_no_batch()
input = torch.randn(100,2)
macs, params = profile(model, inputs=(input, ))
