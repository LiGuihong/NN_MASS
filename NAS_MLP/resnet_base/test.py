#import main


import torch 
import torch.nn as nn 
import torchvision.datasets as dsets 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import random
import numpy as np

from torchvision.models import vgg16
from collections import namedtuple


a= 6
for i in range(3):
    a=a+1
print(a)

'''
x=np.zeros([3,4])
x[0,0]=1
y=np.zeros([3,4])
y[0,0]=1
loss_fn=nn.MSELoss()


x=torch.tensor(x)
y=torch.tensor(y)
print(loss_fn(x,y))
'''




'''
def test_func():
    global bed
    bed =1 
    test_code='bed='+str(5)
    print(test_code)
    exec(test_code,globals())
    print(bed)

    return test_code

a=test_func()


exec(a)
print(bed)

net_depth=main.net_depth

#        self.Q1_3=nn.Parameter(torch.zeros(net_arch[0], net_arch[2]),requires_grad=False)
global str_code
file_code=open('code.txt','w')
print(net_depth)
for i in range(net_depth-2):
    for j in range(i+1):
        str_code='self.Q'+str(j+1)+'_'+str(i+3)+'=nn.Parameter(torch.zeros('+str(main.net_arch[j])+','+str(main.net_arch[i+2])+'),requires_grad=False)'
        print(str_code,file=file_code)




class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer_dict={}
        self.layer_dict['fc0']=nn.Linear(6, 6) # 2 is the dimensions of the data
        self.layer_dict['fc1']=nn.Linear(6, 6)
        self.layer_dict['fc2']=nn.Linear(6, 6)
        self.layer_dict['fc3']=nn.Linear(6, 6)

    def forward(self, x):
        out_dict={}
        out=self.layer_dict['fc0'](x)
        print('asdfasdfasfasdasfdasf')
        out_dict['1']=out
        out1=self.layer_dict['fc1'](out)
        print(out_dict)
        out_dict['2']=out1
        out2=self.layer_dict['fc2'](out1)
        print(out_dict)
        out_dict['3']=out2
        out3=self.layer_dict['fc3'](out2)
        print(out_dict)
        out_dict['3']=out_dict['3']+out_dict['2']
        print(out_dict)
        out_dict['3']=F.relu(out_dict['3'])
        print(out_dict)
        return out
x=torch.Tensor([1,2,3,4,5,6])
model=SimpleNet()
print(model.forward(x))
'''
'''
a=np.ones([4,2])
a=list(a)
a=np.array(a)
print(a)
z=[1,2,3]
a[z,1]=0
print(a[z,:])
tmp=list(np.arange(10))
print(tmp)
print(random.sample(tmp,7))

aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
print(random.sample(aa_milne_arr, 3))

b=link_params=nn.Parameter(torch.zeros([5,2]),requires_grad=False)
b[z,1]=1
print(b)
print(b[z,1])
print(66666666666666666666666)
for x in enumerate(a):
    print(x)

#'''
'''

# argument parser
parser = argparse.ArgumentParser(description='ML_CODESIGN Lab1 - MNIST example')
parser.add_argument('--batch-size', type=int, default=100, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root ='./data',
        train = True,
        transform = transforms.ToTensor(),
        download = True)

test_dataset = dsets.MNIST(root ='./data',
        train = False,
        transform = transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False)
print(test_loader)
# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LogisticRegression(input_size, num_classes)

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# Training the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # (1)
        loss.backward()
        # (2)
        optimizer.step()
        # (3)

        if (i + 1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1,
                       len(train_dataset) // batch_size, loss.data.item()))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    print(images,labels)
    images = Variable(images.view(-1, 28 * 28))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: % d %%' % (100 * correct / total))

'''