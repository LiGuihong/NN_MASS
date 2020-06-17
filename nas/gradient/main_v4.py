import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, datasets
import random
import numpy as np
import argparse

#-------------------------------------
file_logs=open('logs3_v2.txt','a+')
train_logs=open('train3_v2.txt','a+')
# argument parser
parser = argparse.ArgumentParser(description='ML_CODESIGN Lab1 - MNIST example')
parser.add_argument('--batch_size', type=int, default=100, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=200, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--depth', type=int, default=8, help='Learning rate')
parser.add_argument('--width', type=int, default=8, help='Learning rate')
parser.add_argument('--num_seg', type=int, default=2, help='Learning rate')
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
train_num=12000
test_num=1200
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
    np.savetxt("train.csv", np.array(train_data), delimiter=',')
    np.savetxt("test.csv", np.array(test_data), delimiter=',')
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
    train_data = np.loadtxt(open("train.csv","rb"), delimiter=",",dtype=float)
    test_data = np.loadtxt(open("test.csv","rb"), delimiter=",",dtype=float)
    train_label=np.zeros(train_num,dtype=int)
    test_label=np.zeros(test_num,dtype=int)
    for i in range(train_num):
        idx=int(i/(train_num/num_seg))
        train_label[i]=int(idx % 2)
    np.savetxt("tr.csv", np.array(train_label), delimiter=',')
    for i in range(test_num):
        idx=int(i/(test_num/num_seg))
        test_label[i]=int(idx % 2)
    np.savetxt("te.csv", np.array(test_label), delimiter=',')
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


iteration=0

print('    ,'+str(nn_mass) +', '+str(net_width)+', '+str(net_depth)+', '+str(density),file=train_logs) 

for iter_times in range(1):
    model = SimpleNet()
    writer_name='logs/d'+str(net_depth)+'_s'+str(num_seg)+'_l'+str(shortcut_num)+'_w'+str(net_width)+'_m'+str(int(nn_mass))+'__'+str(iter_times)
    writer=SummaryWriter(writer_name)
    def fc0_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc0_layer', output_tensor[0], iteration, bins="auto")
    def fc1_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc1_layer', output_tensor[0], iteration, bins="auto")
    def fc2_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc2_layer', output_tensor[0], iteration, bins="auto")
    def fc3_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc3_layer', output_tensor[0], iteration, bins="auto")
    def fc4_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc4_layer', output_tensor[0], iteration, bins="auto")
    def fc5_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc5_layer', output_tensor[0], iteration, bins="auto")
    def fc6_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc6_layer', output_tensor[0], iteration, bins="auto")
    def fc7_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc7_layer', output_tensor[0], iteration, bins="auto")
    def fc8_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc8_layer', output_tensor[0], iteration, bins="auto")
    def fc9_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc9_layer', output_tensor[0], iteration, bins="auto")
    def fc10_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc10_layer', output_tensor[0], iteration, bins="auto")
    def fc11_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc11_layer', output_tensor[0], iteration, bins="auto")
    def fc12_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc12_layer', output_tensor[0], iteration, bins="auto")
    def fc13_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc13_layer', output_tensor[0], iteration, bins="auto")
    def fc14_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc14_layer', output_tensor[0], iteration, bins="auto")
    def fc15_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc15_layer', output_tensor[0], iteration, bins="auto")
    def fc16_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc16_layer', output_tensor[0], iteration, bins="auto")
    def fc17_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc17_layer', output_tensor[0], iteration, bins="auto")
    def fc18_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc18_layer', output_tensor[0], iteration, bins="auto")
    def fc19_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc19_layer', output_tensor[0], iteration, bins="auto")
    def fc20_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc20_layer', output_tensor[0], iteration, bins="auto")
    def fc21_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc21_layer', output_tensor[0], iteration, bins="auto")
    def fc22_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc22_layer', output_tensor[0], iteration, bins="auto")
    def fc23_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc23_layer', output_tensor[0], iteration, bins="auto")
    def fc24_hook(module, input_tensor, output_tensor):
        writer.add_histogram('fc24_layer', output_tensor[0], iteration, bins="auto")
    model = SimpleNet()

    model.features[0].register_backward_hook(fc0_hook)
    model.features[1].register_backward_hook(fc1_hook)
    model.features[2].register_backward_hook(fc2_hook)
    if net_depth>3:
        model.features[3].register_backward_hook(fc3_hook)   
    if net_depth>4:
        model.features[4].register_backward_hook(fc4_hook)     
    if net_depth>5:
        model.features[5].register_backward_hook(fc5_hook)
    if net_depth>6:
        model.features[6].register_backward_hook(fc6_hook)
    if net_depth>7:
        model.features[7].register_backward_hook(fc7_hook)
    if net_depth>8:
        model.features[8].register_backward_hook(fc8_hook)
    if net_depth>9:
        model.features[9].register_backward_hook(fc9_hook)
    if net_depth>10:
        model.features[10].register_backward_hook(fc10_hook)
    if net_depth>11:
        model.features[11].register_backward_hook(fc11_hook)
    if net_depth>12:
        model.features[12].register_backward_hook(fc12_hook)
    if net_depth>13:
        model.features[13].register_backward_hook(fc13_hook)
    if net_depth>14:
        model.features[14].register_backward_hook(fc14_hook)
    if net_depth>15:
        model.features[15].register_backward_hook(fc15_hook)
    if net_depth>16:
        model.features[16].register_backward_hook(fc16_hook)
    if net_depth>17:
        model.features[17].register_backward_hook(fc17_hook)
    if net_depth>18:
        model.features[18].register_backward_hook(fc18_hook)
    if net_depth>19:
        model.features[19].register_backward_hook(fc19_hook)
    if net_depth>20:
        model.features[20].register_backward_hook(fc20_hook)
    if net_depth>21:
        model.features[21].register_backward_hook(fc21_hook)
    if net_depth>22:
        model.features[22].register_backward_hook(fc22_hook)
    if net_depth>23:
        model.features[23].register_backward_hook(fc23_hook)
    if net_depth>24:
        model.features[24].register_backward_hook(fc24_hook)
    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    iteration=0
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
            iteration=iteration+1
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
    writer.close()