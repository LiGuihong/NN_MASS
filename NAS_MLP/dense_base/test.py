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



x=torch.tensor(np.zeros((8,29)))
y=torch.tensor(np.zeros((8,3)))
x=torch.cat((x,y),1)
print(x.size())
print(torch.cat((x,y),1))
idx=torch.tensor([0, 5, 2, 4, 6, 3])
tmp=x[:,idx]
print(tmp.size())
