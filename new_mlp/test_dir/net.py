import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np

'''
i) testing semi-linear activation functions against sigmoids
ii) evaluating different network architectures such as the number of layers
ii) Adding a convolution front end. DANGER this may be a very time-consuming function.
'''

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001
kernel_size=3
#if (torch.cuda.is_available() and args.enable_cuda):
if (torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#device = torch.device('cpu')
# MNIST Dataset (Images and Labels)
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





# Model
class MLP_Net(nn.Module):
    def __init__(self,depth=1,width=10,net_act='sigmoid',upper_bound=1,lower_bound=-1):
        super(MLP_Net, self).__init__()
        self.depth=depth
        self.width=width
        self.act_type=net_act
        self.upper_bound=upper_bound
        self.lower_bound=lower_bound
        layer_list=[]
        layer_list.append(nn.Linear(784, width))
        for i in range(depth):
            layer_list.append(nn.Linear(width,width))  
        layer_list.append(nn.Linear(width,10))
        self.features = nn.ModuleList(layer_list).eval()
    def forward(self, x):
        out=self.features[0](x)
        for i in range(self.depth+1):
            out=self.features[i+1](out)
            if self.act_type=='relu':
                out=F.relu(out)
            if self.act_type=='sigmoid':
                out=F.sigmoid(out)
            if self.act_type=='softmax':
                out=F.softmax(out,1)
            if self.act_type=='semi_linear':
                out=self.semi_linear(out)
            if self.act_type=='tanh':
                out=F.tanh(out)
            if self.act_type=='elu':
                out=F.elu(out)
            if self.act_type=='leaky_relu':
                out=F.leaky_relu(out)
        return out
    def semi_linear(self,x):   
        upper_mat=torch.ones_like(x)
        lower_mat=torch.zeros_like(x)
        x=torch.where(x<self.upper_bound, x, upper_mat)
        x=torch.where(x>self.lower_bound, x, lower_mat)
        return x



class Conv_net(nn.Module):
    def __init__(self,depth=3):
        super(Conv_net, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module("conv1", nn.Conv2d(
            1, 4, kernel_size=kernel_size, stride=1, padding=1))
        #self.features.add_module("bn1", nn.BatchNorm2d(4))
        self.features.add_module("relu1", nn.ReLU(inplace=False))
        self.features.add_module(
            "pool1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module("conv2", nn.Conv2d(
            4, 16, kernel_size=kernel_size, stride=1, padding=1))
        #self.features.add_module("bn2", nn.BatchNorm2d(16))
        self.features.add_module("relu2", nn.ReLU(inplace=False))
        for i in range(depth-3):
            self.features.add_module("conv"+str(i+3), nn.Conv2d(
                16, 16, kernel_size=kernel_size, stride=1, padding=1))
            #self.features.add_module("bn"+str(depth+3), nn.BatchNorm2d(16))
            self.features.add_module("relu"+str(i+3), nn.ReLU(inplace=False))
        self.features.add_module(
            "pool2", nn.MaxPool2d(kernel_size=2, stride=2))
        self.lin1 = nn.Linear(7 * 7 * 16, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.lin1(out)
        return out



criterion = nn.CrossEntropyLoss()
criterion_sum = nn.CrossEntropyLoss(reduction='sum')
def train(images, labels, optimizer, model,input_size=[batch_size,28*28]):
    images = Variable(images.view(input_size)).to(device)
    labels = Variable(labels).to(device)

    optimizer.zero_grad()
    outputs = model(images)
    print(outputs.size())
    print(labels.size())
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss
def test(epoch,test_loader,model,input_size=[batch_size,28*28]):
    correct = 0
    total = 0
    total_loss = 0
    for images, labels in test_loader:        
        images = Variable(images.view(input_size)).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        total_loss += criterion_sum(outputs, labels)
    accuracy = correct.float() / total
    avg_loss = total_loss / total
    #print('Accuracy of the model on the 10000 test images: % f %%' %
    #    (100 * accuracy))
    return accuracy,avg_loss

'''
net_depth=(np.arange(6,dtype=int)*2+1)
conv_acc=np.zeros(net_depth.size)
conv_loss=np.zeros(net_depth.size)
for idx in range(net_depth.size):
    model=Conv_net(depth=net_depth[idx]).to(device)
    print(model.features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iteration = 0
    # Training the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            loss = train(images, labels,optimizer=optimizer,model=model,input_size=[batch_size,1,28,28])
            if (i + 1) % 100 == 0:
                print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1,
                        len(train_dataset) // batch_size, loss.data.item()))
            iteration += 1
        #test(epoch)
        print(test(0,test_loader=test_loader,model=model,input_size=[batch_size,1,28,28]))
    conv_acc[idx],conv_loss[idx]=test(0,test_loader=test_loader,model=model,input_size=[batch_size,1,28,28])
np.savetxt('conv_acc.csv', conv_acc, delimiter=',')
np.savetxt('conv_loss.csv', conv_loss, delimiter=',')
'''


# Model hyper-parameter
# number of hidden layers
net_depth=(np.arange(6,dtype=int)*2+1)
#number of neurons in hidden layers
net_width=(np.arange(16,dtype=int)*2+10)
#type of activation functions
net_act=['tanh']#,'elu','leaky_relu']
for act_type in net_act:
    acc_name=act_type+'_v1_acc.csv'
    loss_name=act_type+'_v1_loss.csv'
    tmp_acc=np.zeros([net_depth.size,net_width.size])
    tmp_loss=np.zeros([net_depth.size,net_width.size])
    for depth in range(1):
        for width in range(1):
            if (torch.cuda.is_available()):
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            model = MLP_Net(depth=net_depth[depth],width=net_width[width],net_act=act_type).to(device)
            # Loss and Optimizer
            # Softmax is internally computed.
            # Set parameters to be updated.
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            iteration = 0
            # Training the Model
            for epoch in range(num_epochs):
                for i, (images, labels) in enumerate(train_loader):
                    loss = train(images, labels,optimizer=optimizer,model=model)
                    #if (i + 1) % 100 == 0:
                    #    print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    #        % (epoch + 1, num_epochs, i + 1,
                    #            len(train_dataset) // batch_size, loss.data.item()))
                    iteration += 1
                #test(epoch)
            tmp_acc[depth,width],tmp_loss[depth,width]=test(0,test_loader=test_loader,model=model)
    np.savetxt(acc_name, tmp_acc, delimiter=',')
    np.savetxt(loss_name, tmp_loss, delimiter=',')


