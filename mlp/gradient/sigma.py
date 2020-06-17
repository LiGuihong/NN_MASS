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
import collections
import glob
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures 
from scipy.stats import pearsonr
def simulate_m(w,m):
    normal_mat=np.random.normal(size=(w,w+int(m/2)))
    _,sigma,_=np.linalg.svd(normal_mat)
    return np.mean(sigma)


#-------------------------------------
file_logs=open('logs/log_sigma_v0.txt','a+')
train_logs=open('logs/train_sigma_v0.txt','a+')
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
'''
net_depth=int(random.uniform(4,10))
num_all_path=int(((net_depth-2)*(net_depth-1))/2)
shortcut_num=int(random.uniform(0,num_all_path))
'''

net_depth=args.depth
net_width=args.width

learning_rate=args.lr
num_epochs=args.epochs
batch_size=args.batch_size
num_seg=args.num_seg
tc=args.tc
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

#print(net_depth)
#print(all_path_num)
#print(nn_mass)
#exit(0)
class SimpleNet_order_no_batch(torch.nn.Module):
    def __init__(self,act='relu',network_depth=10,network_width=8,tc=0):
        super(SimpleNet_order_no_batch, self).__init__()
        self.depth=network_depth
        self.width=network_width
        self.act=act       
        self.tc=tc
        all_path_num=np.zeros(self.depth)
        layer_tc=np.zeros(self.depth)
        for i in range(self.depth-2):
            for j in range(i+1):
                all_path_num[i+2]=all_path_num[i+2]+self.width
            layer_tc[i+2]=min(tc,all_path_num[i+2])

        self.layer_tc=np.array(layer_tc,dtype=int)
        self.all_path_num=np.array(all_path_num,dtype=int)
        self.density=(np.sum(layer_tc))/(np.sum(all_path_num))
        self.nn_mass=self.density*self.width*self.depth

        batchnorm_list=[]
        layer_list=[]
        layer_list.append(nn.Linear(2, self.width+self.layer_tc[0]))
        batchnorm_list.append(nn.BatchNorm1d(self.width+self.layer_tc[0]))
        
        for i in range(self.depth-2):
            layer_list.append(nn.Linear(self.width+self.layer_tc[i+1], self.width))
            batchnorm_list.append(nn.BatchNorm1d(self.width+self.layer_tc[i+1]))  
        layer_list.append(nn.Linear(self.width+self.layer_tc[self.depth-1], 2))
        self.features = nn.ModuleList(layer_list).eval() 
        self.batchnorm = nn.ModuleList(batchnorm_list).eval()
        self.link_dict=[]
        for i in range(self.depth):
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
        tmp=list((np.arange(self.all_path_num[idx])))
        link_idx=random.sample(tmp,self.layer_tc[idx])
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
        for layer_idx in range(self.depth-2):
            in_features=feat_dict[layer_idx]
            if layer_tc[layer_idx+2]>0:
                in_tmp=torch.cat((out_dict[layer_idx+1],in_features[:,self.link_dict[layer_idx+2]]),1)
                if layer_idx<self.depth-3:
                    out_tmp=self.features[layer_idx+2](F.relu(in_tmp))
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp=self.features[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)           
            else:
                in_tmp=out_dict[layer_idx+1]
                if layer_idx<self.depth-3:
                    out_tmp=self.features[layer_idx+2](F.relu(in_tmp))
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp=self.features[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)     
        return out_dict[self.depth-1]


    def dev(self,fc_layer,layer_idx,x):
        batch_num=x.size()[0]
        if self.act=='relu' and layer_idx>1:
            w=fc_layer.weight.data
            dims=w.size()
            W=w.unsqueeze(0).repeat(batch_num,1,1)
            zero = torch.zeros_like(x)
            one = torch.ones_like(x)
            relu_dev=torch.where(x > 0, one, zero)
            b=torch.zeros([batch_num,dims[1],dims[1]])
            #print(relu_dev.size())
            #print(w.size())
            for i in range(batch_num):
                for j in range(dims[1]):
                    b[i,j,j]=relu_dev[i,j]
            j=torch.matmul(W,b)
            _,sigma,_=torch.svd(j)
            return sigma
        
        if self.act=='elu' and layer_idx>1:
            w=fc_layer.weight.data
            dims=w.size()
            W=w.unsqueeze(0).repeat(batch_num,1,1)
            zero = torch.zeros_like(x)
            one = torch.exp(x)
            relu_dev=torch.where(x > 0, one, zero)
            b=torch.zeros([batch_num,dims[1],dims[1]])
            for i in range(batch_num):
                for j in range(dims[1]):
                    b[i,j,j]=relu_dev[i,j]
            j=torch.matmul(W,b)
            _,sigma,_=torch.svd(j)
            return sigma

        if self.act=='semi_linear' and layer_idx>1:
            w=fc_layer.weight.data
            dims=w.size()
            W=w.unsqueeze(0).repeat(batch_num,1,1)
            zero = torch.zeros_like(x)
            one = torch.ones_like(x)
            relu_dev=torch.where((x <1) and (x >-1), one, zero)
            b=torch.zeros([batch_num,dims[1],dims[1]])
            for i in range(batch_num):
                for j in range(dims[1]):
                    b[i,j,j]=relu_dev[i,j]
            j=torch.matmul(W,b)
            _,sigma,_=torch.svd(j)
            return sigma

        if layer_idx==1:
            w=fc_layer.weight.data
            dims=w.size()
            #print(dims)
            W=w.unsqueeze(0).repeat(batch_num,1,1)
            _,sigma,_=torch.svd(W)
            return sigma
        if layer_idx==0:
            w=fc_layer.weight.data
            dims=w.size()
            #print(dims)
            W=w.unsqueeze(0).repeat(batch_num,1,1)
            zero = torch.zeros_like(x)
            if self.act=='semi_linear':            
                one = torch.ones_like(x)
                relu_dev=torch.where((x <1) and (x >-1), one, zero)
            if self.act=='relu':            
                one = torch.ones_like(x)
                relu_dev=torch.where(x > 0, one, zero)
            if self.act=='elu':            
                one = torch.exp(x)
                relu_dev=torch.where(x > 0, one, zero)

            b=torch.zeros([batch_num,dims[0],dims[0]])
            for i in range(batch_num):
                for j in range(dims[0]):
                    b[i,j,j]=relu_dev[i,j]
            #print(W.size())
            #print(b.size())
            j=torch.matmul(b,W)
            _,sigma,_=torch.svd(j)
            return sigma

    def isometry(self, x):
        out0=F.relu(self.features[0](x))        
        out1=self.features[1](out0) 

        in_dict=[]
        in_dict.append(self.features[0](x))
        in_dict.append(out0)

        out_dict=[]
        out_dict.append(out0)
        out_dict.append(out1)
        feat_dict=[]
        feat_dict.append(out0)
        feat_dict.append(torch.cat((out1,out0),1))
        for layer_idx in range(net_depth-2):
            in_features=feat_dict[layer_idx]
            if self.layer_tc[layer_idx+2]>0:
                in_tmp=torch.cat((out_dict[layer_idx+1],in_features[:,self.link_dict[layer_idx+2]]),1)
                in_dict.append(in_tmp)
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
                in_dict.append(in_tmp)
                if layer_idx<net_depth-3:
                    out_tmp=self.features[layer_idx+2](F.relu(in_tmp))
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp=self.features[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)  
        sigma_all=None
        layer_sigma=[]
        for i in range(net_depth):
            sigma=self.dev(self.features[i],layer_idx=i,x=in_dict[i]) 
            layer_sigma.append(sigma.view(-1).mean())
            if i==0:
                sigma_all=sigma
            else:
                #print('-------------')
                #print(sigma_all.size())
                #print(sigma.size())
                sigma_all=torch.cat((sigma_all,sigma),1)
        
        sig_mean=sigma_all.view(-1).mean()
        sig_std=sigma_all.view(-1).std()
        return sig_mean,sig_std,layer_sigma

model = SimpleNet_order_no_batch()

def make_dataset(train_file='data/train_circle.csv',test_file='data/test_circle.csv'):
    train_data = np.loadtxt(open(train_file,"rb"), delimiter=",",dtype=float)
    test_data = np.loadtxt(open(test_file,"rb"), delimiter=",",dtype=float)
    train_label=np.zeros(train_num,dtype=int)
    test_label=np.zeros(test_num,dtype=int)
    for i in range(train_num):
        idx=int(i/(train_num/num_seg))
        train_label[i]=int(idx % 2)
    np.savetxt("tr_lin.csv", np.array(train_label), delimiter=',')
    for i in range(test_num):
        idx=int(i/(test_num/num_seg))
        test_label[i]=int(idx % 2)
    np.savetxt("te_lin.csv", np.array(test_label), delimiter=',')
    return train_data,test_data,train_label.T,test_label.T

[train_data_raw,test_data,train_label_raw,test_label]=make_dataset()
raw_label=np.zeros([train_num,2])
for i in range(train_num):
    raw_label[i,train_label_raw[i]]=1
train_data_raw=torch.tensor(train_data_raw,dtype=torch.float)
test_data=torch.tensor(test_data,dtype=torch.float)
test_data_raw=test_data
train_label_raw=Variable(torch.from_numpy(train_label_raw))
test_label=Variable(torch.from_numpy(test_label))
test_label_raw=test_label

#print('*************************************************************',file=file_logs)
#print('*************************************************************',file=file_logs)

#print('*************************************************************',file=train_logs)
#print('*************************************************************',file=train_logs)

def find_bound(predicted):
    bound=[]
    bound_num=0
    for i in range(test_num-1):
        if predicted[i]!=predicted[i+1]:
            bound.append(i)
            bound_num+=1
    step=int(min(bound_num,num_seg-1))

    return bound_num,bound


#---------------------for test----------------------
'''
model = SimpleNet_order_no_batch()
print(model.features)
#print(model.link_dict)
#print(model.li)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer.zero_grad()
outputs = model.isometry(train_data_raw)
print(outputs)


exit(0)
'''
mean_file='sigma/sigma_mean_'+str(net_width)+'_'+str(net_depth)+'_'+str(tc)+'_'+str(num_seg)+'.csv'
std_file='sigma/sigma_std_'+str(net_width)+'_'+str(net_depth)+'_'+str(tc)+'_'+str(num_seg)+'.csv'
acc_file='sigma/acc_test_'+str(net_width)+'_'+str(net_depth)+'_'+str(tc)+'_'+str(num_seg)+'.csv'
sigma_file='sigma/all_sigma_'+str(net_width)+'_'+str(net_depth)+'_'+str(tc)+'_'+str(num_seg)+'.csv'
train_file='sigma/acc_train_'+str(net_width)+'_'+str(net_depth)+'_'+str(tc)+'_'+str(num_seg)+'.csv'

mean_sigma=[]
std_sigma=[]
acc_test=[]
acc_train=[]
layer_mat=[]






trial_index=0
width_num=5
depth_num=4
tc_num=15
legend=[]
nn_mass_mat=np.zeros([depth_num,tc_num])
sig_mean_mat=np.zeros([depth_num,tc_num])
params_mat=np.zeros([depth_num,tc_num])

for j in range(depth_num): #depth  16,20,24 28
    legend.append(str(16+4*j)+'-layers')
    for k in range(tc_num):  #tc 0 2 4 6 8 10 12 
        for iter_times in range(10):
            model = SimpleNet_order_no_batch(network_depth=16+8*j,network_width=8,tc=k)
            #print(model.features)
            #print(model.link_dict)
            #print(model.li)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
            print(datetime.datetime.now())
            sig_mean,_,_=model.isometry(test_data)

            sig_mean_mat[j,k]=sig_mean_mat[j,k]+sig_mean
            nn_mass_mat[j,k]=model.nn_mass
            params_mat[j,k]=model.param_num()[0]
        sig_mean_mat[j,k]=sig_mean_mat[j,k]/10

np.savetxt('sigmean.txt',sig_mean_mat)
font_size=14

X=nn_mass_mat.reshape(-1,1) 
y=sig_mean_mat.reshape(-1,1)
reg2 = LinearRegression()
reg2.fit(X, y)
score = reg2.score(X,y)
plt.figure(1)
plt.title('R^2='+str(score))

plt.plot(nn_mass_mat[0,:],sig_mean_mat[0,:],'-s')
plt.plot(nn_mass_mat[1,:],sig_mean_mat[1,:],'-o')
plt.plot(nn_mass_mat[2,:],sig_mean_mat[2,:],'-^')
plt.plot(nn_mass_mat[3,:],sig_mean_mat[3,:],'-v')
plt.plot(X,reg2.predict(X),'p')
plt.grid(linestyle=':')
plt.xlabel('NN-Mass',fontsize=font_size)
plt.ylabel('Initial Singular Value (Mean)',fontsize=font_size)
plt.legend(['16-layers','20-layers','24-layers','28-layers','linear regression'],fontsize=font_size)

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.title('Initial Singular Value vs. NN-Mass ($R^2$='+str(round(score, 2))+')',fontsize=font_size+2)
plt.savefig('fig/nnmass_sigma_seg20.svg',format='svg')
plt.savefig('fig/nnmass_sigma_seg20.png',format='png')



X=params_mat.reshape(-1,1) 
reg2 = LinearRegression()
reg2.fit(X, y)
score = reg2.score(X,y)
plt.figure(2)
plt.title('R^2='+str(score))
plt.plot(params_mat[0,:],sig_mean_mat[0,:],'-s')
plt.plot(params_mat[1,:],sig_mean_mat[1,:],'-o')
plt.plot(params_mat[2,:],sig_mean_mat[2,:],'-^')
plt.plot(params_mat[3,:],sig_mean_mat[3,:],'-v')

plt.plot(X,reg2.predict(X),'p')
plt.grid(linestyle=':')
plt.xlabel('Number of Parameters',fontsize=font_size)
plt.ylabel('Initial Singular Value (Mean)',fontsize=font_size)
plt.legend(['16-layers','20-layers','24-layers','28-layers','linear regression'],fontsize=font_size)

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.title('Initial Singular Value vs. Number of Params ($R^2$='+str(round(score, 2))+')',fontsize=font_size+2)
plt.savefig('fig/params_sigma_seg20.svg',format='svg')
plt.savefig('fig/params_sigma_seg20.png',format='png')









'''
trial_index=0
width_num=5
depth_num=3
tc_num=7
for i in range(width_num): #width 8,16,32,64,128
    plt.figure(i)
    legend=[]
    nn_mass_mat=np.zeros([depth_num,tc_num])
    sig_mean_mat=np.zeros([depth_num,tc_num])
    sim_nn_mass=np.zeros([depth_num,tc_num])
    for j in range(depth_num): #depth  16,20,24
        legend.append(str(16+4*j)+'-layers')
        for k in range(tc_num):  #tc 0 2 4 6 8 10 12 
            for iter_times in range(3):
                model = SimpleNet_order_no_batch(network_depth=16+8*j,network_width=pow(2,i+3),tc=2*k)
                #print(model.features)
                #print(model.link_dict)
                #print(model.li)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
                print(datetime.datetime.now())
                sig_mean,sig_std,layer_sigma=model.isometry(train_data_raw)
                layer_mat.append(layer_sigma)
                mean_sigma.append(sig_mean)
                std_sigma.append(sig_std)
                sig_mean_mat[j,k]=sig_mean_mat[j,k]+sig_mean
                nn_mass_mat[j,k]=model.nn_mass
                sim_nn_mass[j,k]=sim_nn_mass[j,k]+simulate_m(w=pow(2,i+3),m=model.nn_mass)
            sig_mean_mat[j,k]=sig_mean_mat[j,k]/3
            sim_nn_mass[j,k]=sim_nn_mass[j,k]/3

        plt.plot(nn_mass_mat[j,:],sig_mean_mat[j,:],'-o')
        print(888)
    fig_name='fig/' + 'width_'+str(pow(2,i+3))+'_Mean_Singular_Value_with_NN_MASS_'+str(trial_index)
    plt.title(fig_name)
    plt.xlabel('NN-MASS')
    plt.ylabel('Mean Singular Value')
    plt.legend(legend)
    plt.savefig(fig_name+'.png',ppi=600,format='png')
    print(666)
    plt.figure(i+40)
    plt.plot(nn_mass_mat.reshape(-1),sim_nn_mass.reshape(-1),'o')
    plt.xlabel('NN-MASS')
    plt.ylabel('Mean Singular Value')


    X=nn_mass_mat.reshape(-1,1) 
    y=sim_nn_mass.reshape(-1,1)
    reg2 = LinearRegression()
    reg2.fit(X, y)
    score = reg2.score(X,y)
    plt.figure(1)
    plt.title('Widht='+str(pow(2,i+3))+' R^2='+str(score))
    plt.plot(X,reg2.predict(X),'-p')
    plt.legend(['Ground Truth','Linear Gegression'])
    plt.savefig(fig_name+'_sim.png',ppi=600,format='png')
    print(666)

'''


