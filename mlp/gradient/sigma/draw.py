import glob
import os
import numpy as np
import matplotlib.pyplot as plt

nn_mass=[]
std_mat=[]
paths = glob.glob(os.path.join('./', 'sigma_init*'))
for file_name in paths:
    res=file_name.split('_')
    data=np.loadtxt(open(file_name,"rb"), delimiter=",",dtype=float)
    nn_mass.append(float(res[5].replace('.csv', '')))
    std_mat.append(np.std(data))
print(nn_mass)
print(std_mat)
print(res[5].replace('.csv', ''))