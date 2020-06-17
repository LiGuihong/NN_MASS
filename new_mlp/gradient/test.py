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

'''
a=torch.zeros([100,16,8])
b=torch.zeros([100,8,8])
c=torch.matmul(a,b)
print(c.size())
_,sigma,_=torch.svd(c)
print(sigma.size())
x=torch.ones([4,2])

print(y.size())
'''
