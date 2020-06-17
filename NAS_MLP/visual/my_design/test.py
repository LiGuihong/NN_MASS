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


x=torch.empty(3, 6).random_(30)
print(x)
u,sig,v=torch.svd(x)
print(u.size())
print(sig)
print(v.size())
print('----------------')

tmp=torch.matmul(u,torch.diag(sig))
print(x.size())
print(tmp.size())
print(u.size())
print(torch.diag(sig).size())
print(v.size())
tmp_v=torch.transpose(v,0,1)
tmp=torch.matmul(tmp,tmp_v)
print(tmp)
x=torch.matmul(tmp,v)