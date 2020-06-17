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

# argument parser
parser = argparse.ArgumentParser(description='ML_CODESIGN Lab1 - MNIST example')
parser.add_argument('--batch_size', type=int, default=100, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--depth', type=int, default=3, help='Learning rate')
parser.add_argument('--num_seg', type=int, default=2, help='Learning rate')
args = parser.parse_args()

learning_rate=args.lr
num_epochs=args.epoches
batch_size=args.batch_size
'''
Hyper-parameter definition
'''
#define the scaling factor for the network's width
arch_scale=0.5

#define the depth of the network
net_depth=args.depth
num_all_path=int(((net_depth-2)*(net_depth-1))/2)
layer_density=np.zeros(net_depth-2)
short_cut_num_record=np.ones(net_depth-2) #the number of layer's connection
short_cut_record=np.zeros([net_depth-2,net_depth])# the index between some layer with former layer
#the number of the decision district, start with 2
num_seg=args.sum_seg #
range_min=0
range_max=1
train_num=12000
test_num=1200
max_category=120

#define the number of neuron in each FC layer

net_arch=[32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32]
net_name=['fc1','fc2','fc3','fc4','fc5','fc6','fc7','fc8','fc9','fc10','fc11','fc12','fc13','fc14','fc15','fc16','fc17','fc18','fc19']

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




# step1: 定义MyDataset类， 继承Dataset, 重写抽象方法：__len()__, __getitem()__
class train_dst(Dataset):
    
    def __init__(self, transform=None):
        self.transform = transform
        self.size = train_num

    def __len__(self):
        return self.size

    def __getitem__(self):
        train_data = np.loadtxt(open("train.csv","rb"), delimiter=",")
        train_label=np.zeros(train_num,dtype=int)
        for i in range(train_num):
            idx=int(i/(train_num/num_seg))
            train_label[i]=int(idx % 2)        
        sample = {'image': train_data, 'label': train_label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class test_dst(Dataset):    
    def __init__(self, transform=None):
        self.transform = transform
        self.size = test_num

    def __len__(self):
        return self.size

    def __getitem__(self):

        test_data = np.loadtxt(open("test.csv","rb"), delimiter=",")
        test_label=np.zeros(test_num,dtype=int)
        for i in range(test_num):
            idx=int(i/(test_num/num_seg))
            test_label[i]=int(idx % 2)
        sample = {'image': test_data, 'label': test_label}
        if self.transform:
            sample = self.transform(sample)
        return sample

# [1,3] [1,4][2,4] [1,5][2,5][3,5] 
# [1,6][2,6][3,6][4,6] 
# [1,7][2,7][3,7][4,7][5,7]
train_dataset=train_dst()
test_dataset=test_dst() 

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

test_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)

def shortcut_nn_mass():
    if net_depth<3:
        return None
    else:
        for i in range(net_depth-2):
            for j in range(i):
                short_path=[j+1,i+3]
                all_short_cut.append(short_path);
    all_path_num=np.zeros(net_depth-2)
    for i in range(net_depth-2):
        for j in range(i+1):
            all_path_num[i]=all_path_num[i]+net_arch[i+2]*(net_arch[j])


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
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        self.layer_dict={}
        self.layer_dict[net_name[0]]=nn.Linear(2, net_arch[0])        
        for i in range(net_depth-2):
            self.layer_dict[net_name[i+1]]=nn.Linear(net_arch[i], net_arch[i+1])
        self.layer_dict[net_name[net_depth-1]]=nn.Linear(net_arch[net_depth-1], 2)

        
        self.link_dict={}
        for i in range(shortcut_num):
            link_name='l'+str(short_cut[i,0])+'_'+str(short_cut[i,1])
            self.link_dict[link_name]=self.add_link(i)


    def add_link(self,idx=0):
        link_params=nn.Parameter(torch.zeros(net_arch[short_cut[idx,0]],net_arch[short_cut[idx,1]]),requires_grad=False)
        for i in range(net_arch[short_cut[idx,1]]):
            link_idx=random.choice(np.arange(net_arch[short_cut[idx,0]]),shortcut_tc[idx])
            link_params[:,i]=link_idx
        return link_params


    def forward(self, x):
        out1=F.relu(self.fc1(x))        
        out2=F.relu(self.fc2(out1))
        out_dict={}
        out_dict['1']=out1
        out_dict['2']=out2
        
        for layer_idx in range(net_depth-2):
            layer_name='fc'+str(layer_idx+3)
            out_dict[str(layer_idx+3)]=self.layer_dict[layer_name](out_dict[str(layer_idx+2)])
            for k in range(short_cut_num_record[layer_idx]):
                link_name='l'+str(short_cut_record[layer_idx,0])+'_'+str(layer_idx+3)
                out_dict[str(layer_idx+3)]=out_dict[str(layer_idx+3)]+out_dict[str(short_cut_record[layer_idx,0]+1)]*self.link_dict[link_name]
            out_dict[str(layer_idx+3)]=F.relu(out_dict[str(layer_idx+3)])

        return out_dict[str(net_depth)]





model = SimpleNet()


# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


    

for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (image, label) in enumerate(train_dataloader):
        image = Variable(image)
        label = Variable(label)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        # (1)
        loss.backward()
        # (2)
        optimizer.step()
        # (3)

        if (i + 1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1,
                       len(train_dataset) // batch_size, loss.data.item()))
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum()
    print('Train Accuracy after epoch '+str(epoch+1) +' ,' +str(100 * correct / total))

# Test the Model
correct = 0
total = 0

for image, label in test_dataloader:
    image = Variable(image)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    total += label.size(0)
    correct += (predicted == label).sum()

print('Accuracy of the model on the 10000 test images: % d %%' % (100 * correct / total))