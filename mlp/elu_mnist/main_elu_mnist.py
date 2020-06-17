import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, datasets
import random
import numpy as np
import argparse
import datetime
import collections
#-------------------------------------
file_logs=open('logs/log_elu_mnist_v3.txt','a+')
train_logs=open('logs/train_elu_mnist_v3.txt','a+')
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
input_size = 784
num_classes = 10

kernel_size=3
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

train_num=60000
test_num=10000
if (torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

device = torch.device('cpu')
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#define the number of neuron in each FC layer

net_arch=[8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8]
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
    def __init__(self,act='relu'):
        super(SimpleNet_order_no_batch, self).__init__()
        batchnorm_list=[]
        layer_list=[]
        layer_list.append(nn.Linear(784, net_arch[0]+layer_tc[0]))
        batchnorm_list.append(nn.BatchNorm1d(net_arch[0]+layer_tc[0]))
        self.depth=net_depth
        self.act=act
        for i in range(net_depth-2):
            layer_list.append(nn.Linear(net_arch[i+1]+layer_tc[i+1], net_arch[i+1]))
            batchnorm_list.append(nn.BatchNorm1d(net_arch[i+1]+layer_tc[i+1]))  
        layer_list.append(nn.Linear(net_arch[net_depth-1]+layer_tc[net_depth-1], 10))
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
        out0=F.elu(self.features[0](x))        
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
                    out_tmp=self.features[layer_idx+2](F.elu(in_tmp))
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp=self.features[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)           
            else:
                in_tmp=out_dict[layer_idx+1]
                if layer_idx<net_depth-3:
                    out_tmp=self.features[layer_idx+2](F.elu(in_tmp))
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)
                else:
                    out_tmp=self.features[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict.append(out_tmp)     
        return out_dict[net_depth-1]


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
            if layer_tc[layer_idx+2]>0:
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
        for i in range(net_depth):
            sigma=self.dev(self.features[i],layer_idx=i,x=in_dict[i])   
            if i==0:
                sigma_all=sigma
            else:
                #print('-------------')
                #print(sigma_all.size())
                #print(sigma.size())
                sigma_all=torch.cat((sigma_all,sigma),1)
        
        sig_mean=sigma_all.view(-1).mean()
        sig_std=sigma_all.view(-1).std()
        return sig_mean,sig_std

model = SimpleNet_order_no_batch()


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


criterion = nn.CrossEntropyLoss()
criterion_sum = nn.CrossEntropyLoss(reduction='sum')
def train(images, labels, optimizer, model,input_size=[batch_size,28*28]):
    images = Variable(images.view(input_size)).to(device)
    labels = Variable(labels).to(device)

    optimizer.zero_grad()
    outputs = model(images)
    #print(outputs.size())
    #print(labels.size())
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss,outputs
def test(epoch,test_loader,model,input_size=[batch_size,28*28]):
    correct = 0
    total = 0
    total_loss = 0
    i=0
    predicted_label=torch.zeros(test_num)
    for images, labels in test_loader:        
        images = Variable(images.view(input_size)).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        total_loss += criterion_sum(outputs, labels)
        predicted_label[i*batch_size:(i+1)*batch_size]=predicted
        i=i+1
    accuracy = correct.float() / total
    avg_loss = total_loss / total
    #print('Accuracy of the model on the 10000 test images: % f %%' %
    #    (100 * accuracy))
    return accuracy,avg_loss
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

'''
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
mean_file='sigma/sigma_mean_'+str(net_width)+'_'+str(net_depth)+'_'+str(tc)+'_'+str(num_seg)+'.csv'
std_file='sigma/sigma_std_'+str(net_width)+'_'+str(net_depth)+'_'+str(tc)+'_'+str(num_seg)+'.csv'
acc_file='sigma/acc_test_'+str(net_width)+'_'+str(net_depth)+'_'+str(tc)+'_'+str(num_seg)+'.csv'
mean_sigma=[]
std_sigma=[]
acc_test=[]
'''

for iter_times in range(5):
    model = SimpleNet_order_no_batch().to(device)
    #print(model.features)
    #print(model.link_dict)
    #print(model.li)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    print(datetime.datetime.now())
    max_acc=-1000
    for epoch in range(num_epochs):

        correct = 0
        total = 0
        total_loss=0  
        for i, (images, labels) in enumerate(train_loader):
            loss,outputs = train(images, labels,optimizer=optimizer,model=model)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            total_loss=total_loss+loss
        total_loss=total_loss/float(i+1)
        print(str(100 * float(correct) / total)+','+str(total_loss),file=train_logs)


    tmp_acc,tmp_loss=test(0,test_loader=test_loader,model=model)
#    print(str(100 * tmp_acc)+','+str(tmp_loss),file=train_logs)
    print(str(nn_mass) +', '+str((100 * tmp_acc))+', '
        +str(model.params)+', '+str(model.flops)+', '+str(net_width)+', '+str(net_depth)+', '
        +str(tc)+', '+str(num_seg)+', ',file=file_logs) 
    print('Accuracy of the model on the 10000 test images: % f %%' % (100 * tmp_acc))
'''
for iter_times in range(5):
    model = SimpleNet_order_no_batch().to(device)
    #print(model.features)
    #print(model.link_dict)
    #print(model.li)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    print(datetime.datetime.now())
    max_acc=-1000
    for epoch in range(num_epochs):

        correct = 0
        total = 0
        total_loss=0  
        for i, (images, labels) in enumerate(train_loader):
            loss,outputs = train(images, labels,optimizer=optimizer,model=model)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            total_loss=total_loss+loss
        if max_acc<float(correct) / total:
            max_acc=float(correct) / total
        total_loss=total_loss/float(i+1)
        print(str(100 * float(correct) / total)+','+str(total_loss),file=train_logs)
        #print('Train Accuracy after epoch '+str(epoch+1) +' ,' +str(100 * float(correct) / total))
#        if epoch % 20 == 0:
#            sig_mean,sig_std=model.isometry(train_data_raw)
#            mean_sigma.append(sig_mean)
#            std_sigma.append(sig_std)
#
#            outputs = model(test_data_raw)
#            _, predicted = torch.max(outputs.data, 1)
#            correct = (predicted == test_label_raw).sum()
#            acc_test.append(float(correct)/float(test_num))
    #print('------******------******------******------******------******------******------',file=train_logs)
    # Test the Model
    tmp_acc,tmp_loss=test(0,test_loader=test_loader,model=model)

    #print('------******------******------******------******------******------******------',file=file_logs)
    print(str(nn_mass) +', '+str((100 * tmp_acc))+','+str(100*max_acc)+', '
        +str(model.params)+', '+str(model.flops)+', '+str(net_width)+', '+str(net_depth)+', '
        +str(tc)+', '+str(num_seg)+', ',file=file_logs) 
    print('Accuracy of the model on the 10000 test images: % f %%' % (100 * float(correct) / total))
    #print(datetime.datetime.now())
#np.savetxt(mean_file, np.array(mean_sigma), delimiter=',')
#np.savetxt(std_file, np.array(std_sigma), delimiter=',')
#np.savetxt(acc_file, np.array(acc_test), delimiter=',')
#'''

'''
file_logs=open('logs/sigma_value.logs','a+')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


for iter_times in range(5):
    model = SimpleNet_order_no_batch(act='elu').to(device)
    #print(model.features)
    #print(model.link_dict)
    #print(model.li)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    print(datetime.datetime.now())
    sig_mean=0
    sig_std=0

    for i, (images, labels) in enumerate(test_loader):
        sig_mean_tmp,sig_std_tmp=model.isometry(images.view([batch_size,784]))
        sig_mean=sig_mean+sig_mean_tmp
        sig_std=sig_std+sig_std_tmp
        #print(sig_mean)
        #print(i)
    sig_mean=sig_mean/(i+1)
    sig_std=sig_std/(i+1)
    #print('------******------******------******------******------******------******------',file=file_logs)
    print(str(nn_mass) +', '+str(sig_mean)+', '+str(sig_std)+', '+str(model.params)
    +', '+str(model.flops)+', '+str(net_width)+', '+str(net_depth)+', ' +str(tc)+', ',file=file_logs) 
    #print(datetime.datetime.now())
#'''