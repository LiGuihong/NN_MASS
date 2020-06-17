import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures 
debug=True


def csv_process(filename):
    f = open(filename,'r')
    line = f.readline() 
    data=[]
    while line:
        line_data=line.lstrip('[')
        line_data=line_data.replace('\n', '')
        line_raw=line_data.replace(']', '').split(',' )
        data.append(list(map(float, line_raw)))
        line = f.readline()     
    #print(line_raw)
    return np.array(data)
paths = glob.glob(os.path.join('./', 'gra*'))

depth_mat=[16,20]
tc_mat=[20,40,60,80,100]
layer_idx=1
csv_process(paths[0])
fig_idx=1
for depth in depth_mat:
    plt.figure(fig_idx)
    legend=[]
    if debug==True:
        layer_idx=depth-1
    for tc in tc_mat:
        file_sub_name='gra_mean_8_'+str(depth)+'_'+str(tc)+'*'
        print(file_sub_name)
        paths = glob.glob(os.path.join('./', file_sub_name))
        print(paths)
        data_raw=csv_process(paths[0])

        shape=data_raw.shape
        idx=np.arange(5)
        data=np.zeros([int(shape[0]/5),shape[1]])
        for i in range(int(shape[0]/5)):
            data[i,:]=np.mean(data_raw[idx*i,:],axis=0)
        step=int(shape[1]/4)
        iter_times=np.arange(int(shape[0]/5))
        plt.plot(iter_times,data[:,step*1+layer_idx])
        legend.append(str(depth)+'-layers tc='+str(tc))
    plt.legend(legend)
    plt.xlabel('Training epochs')
    plt.ylabel('Mean absolute gradient')
    plt.title(str(depth)+'-layer: layer_idx='+str(layer_idx))
    plt.savefig('fig/mean_abs_keepzero_'+str(depth)+'_'+str(layer_idx)+'.png',format='png')
    fig_idx=fig_idx+1

for depth in depth_mat:
    plt.figure(fig_idx)
    legend=[]
    if debug==True:
        layer_idx=depth-1
    for tc in tc_mat:
        file_sub_name='gra_mean_8_'+str(depth)+'_'+str(tc)+'*'
        print(file_sub_name)
        paths = glob.glob(os.path.join('./', file_sub_name))
        print(paths)
        data_raw=csv_process(paths[0])

        shape=data_raw.shape
        idx=np.arange(5)
        data=np.zeros([int(shape[0]/5),shape[1]])
        for i in range(int(shape[0]/5)):
            data[i,:]=np.mean(data_raw[idx*i,:],axis=0)
        step=int(shape[1]/4)
        iter_times=np.arange(int(shape[0]/5))
        plt.plot(iter_times,data[:,step*3+layer_idx])
        legend.append(str(depth)+'-layers tc='+str(tc))
    plt.legend(legend)
    plt.xlabel('Training epochs')
    plt.ylabel('Mean absolute gradient')
    plt.title(str(depth)+'-layer: layer_idx='+str(layer_idx))
    plt.savefig('fig/mean_abs_nonzero_'+str(depth)+'_'+str(layer_idx)+'.png',format='png')
    fig_idx=fig_idx+1


for depth in depth_mat:
    plt.figure(fig_idx)
    legend=[]
    for tc in tc_mat:
        file_sub_name='gra_mean_8_'+str(depth)+'_'+str(tc)+'*'
        print(file_sub_name)
        paths = glob.glob(os.path.join('./', file_sub_name))
        print(paths)
        data_raw=csv_process(paths[0])

        shape=data_raw.shape
        data=np.zeros([int(shape[0]/5),shape[1]])
        idx=np.arange(5)
        for i in range(int(shape[0]/5)):
            data[i,:]=np.mean(data_raw[idx*i,:],axis=0)
        step=int(shape[1]/4)
        iter_times=np.arange(int(shape[0]/5))
        plt.plot(iter_times,np.mean(data[:,3*step:4*step],axis=1))
        legend.append(str(depth)+'-layers tc='+str(tc))
    plt.legend(legend)
    plt.xlabel('Training epochs')
    plt.ylabel('Mean absolute gradient')
    plt.title(str(depth)+'-layer')
    plt.savefig('fig/mean_abs_nonzero_all'+str(depth)+'_'+'.png',format='png')
    fig_idx=fig_idx+1

for depth in depth_mat:
    plt.figure(fig_idx)
    legend=[]
    for tc in tc_mat:
        file_sub_name='gra_mean_8_'+str(depth)+'_'+str(tc)+'*'
        print(file_sub_name)
        paths = glob.glob(os.path.join('./', file_sub_name))
        print(paths)
        data_raw=csv_process(paths[0])

        shape=data_raw.shape
        data=np.zeros([int(shape[0]/5),shape[1]])
        idx=np.arange(5)
        for i in range(int(shape[0]/5)):
            data[i,:]=np.mean(data_raw[idx*i,:],axis=0)
        step=int(shape[1]/4)
        iter_times=np.arange(int(shape[0]/5))
        plt.plot(iter_times,np.mean(data[:,1*step:2*step],axis=1))
        legend.append(str(depth)+'-layers tc='+str(tc))
    plt.legend(legend)
    plt.xlabel('Training epochs')
    plt.ylabel('Mean absolute gradient')
    plt.title(str(depth)+'-layer')
    plt.savefig('fig/mean_abs_keepzero_all'+str(depth)+'_'+'.png',format='png')
    fig_idx=fig_idx+1


for depth in depth_mat:
    plt.figure(fig_idx)
    legend=[]
    for tc in tc_mat:
        file_sub_name='sparse_8_'+str(depth)+'_'+str(tc)+'*'
        print(file_sub_name)
        paths = glob.glob(os.path.join('./', file_sub_name))
        print(paths)
        data_raw=csv_process(paths[0])

        shape=data_raw.shape
        data=np.zeros([int(shape[0]/5),shape[1]])
        idx=np.arange(5)
        for i in range(int(shape[0]/5)):
            data[i,:]=np.mean(data_raw[idx*i,:],axis=0)
        step=int(shape[1]/2)
        iter_times=np.arange(int(shape[0]/5))
        spasity=np.sum(data[:,1*step:2*step],axis=1)/np.sum(data[:,0*step:1*step],axis=1)
        print(np.sum(data[0,1*step:2*step]))
        print(np.sum(data[1,1*step:2*step]))
        #print(np.sum(data[:,1*step:2*step],axis=1))
        plt.plot(iter_times,spasity)
        legend.append(str(depth)+'-layers tc='+str(tc))
    plt.legend(legend)
    plt.xlabel('Training epochs')
    plt.ylabel('spasity')
    plt.title(str(depth)+'-layer')
    plt.savefig('fig/spasity_all'+str(depth)+'_'+'.png',format='png')
    fig_idx=fig_idx+1


for depth in depth_mat:
    plt.figure(fig_idx)
    legend=[]
    for tc in tc_mat:
        file_sub_name='gra_std_8_'+str(depth)+'_'+str(tc)+'*'
        print(file_sub_name)
        paths = glob.glob(os.path.join('./', file_sub_name))
        print(paths)
        data_raw=csv_process(paths[0])

        shape=data_raw.shape
        data=np.zeros([int(shape[0]/5),shape[1]])
        idx=np.arange(5)
        for i in range(int(shape[0]/5)):
            data[i,:]=np.mean(data_raw[idx*i,:],axis=0)
        step=int(shape[1]/4)
        iter_times=np.arange(int(shape[0]/5))
        plt.plot(iter_times,np.mean(data[:,1*step:2*step],axis=1))
        legend.append(str(depth)+'-layers tc='+str(tc))
    plt.legend(legend)
    plt.xlabel('Training epochs')
    plt.ylabel('Mean absolute gradient')
    plt.title(str(depth)+'-layer')
    plt.savefig('fig/std_abs_keepzero_all'+str(depth)+'_'+'.png',format='png')
    fig_idx=fig_idx+1

for depth in depth_mat:
    plt.figure(fig_idx)
    legend=[]
    for tc in tc_mat:
        file_sub_name='gra_std_8_'+str(depth)+'_'+str(tc)+'*'
        print(file_sub_name)
        paths = glob.glob(os.path.join('./', file_sub_name))
        print(paths)
        data_raw=csv_process(paths[0])

        shape=data_raw.shape
        data=np.zeros([int(shape[0]/5),shape[1]])
        idx=np.arange(5)
        for i in range(int(shape[0]/5)):
            data[i,:]=np.mean(data_raw[idx*i,:],axis=0)
        step=int(shape[1]/4)
        iter_times=np.arange(int(shape[0]/5))
        plt.plot(iter_times,np.mean(data[:,3*step:4*step],axis=1))
        legend.append(str(depth)+'-layers tc='+str(tc))
    plt.legend(legend)
    plt.xlabel('Training epochs')
    plt.ylabel('Mean absolute gradient')
    plt.title(str(depth)+'-layer')
    plt.savefig('fig/std_abs_nonzero_all'+str(depth)+'_'+'.png',format='png')
    fig_idx=fig_idx+1
'''
for depth in depth_mat:
    plt.figure(fig_idx)
    legend=[]
    for tc in tc_mat:
        file_sub_name='gra_mean_8_'+str(depth)+'_'+str(tc)+'*'
        print(file_sub_name)
        paths = glob.glob(os.path.join('./', file_sub_name))
        print(paths)
        data_raw=csv_process(paths[0])

        shape=data_raw.shape
        data=np.zeros([int(shape[0]/5),shape[1]])
        idx=np.arange(5)
        for i in range(int(shape[0]/5)):
            data[i,:]=np.mean(data_raw[idx*i,:],axis=0)
        step=int(shape[1]/4)
        iter_times=np.arange(int(shape[0]/5))
        plt.plot(iter_times,np.mean(data[:,3*step:4*step],axis=1))
        legend.append(str(depth)+'-layers tc='+str(tc))
    plt.legend(legend)
    plt.xlabel('Training epochs')
    plt.ylabel('Mean absolute gradient')
    plt.title(str(depth)+'-layer')
    plt.savefig('fig/mean_abs_nonzero_all'+str(depth)+'_'+'.png',format='png')
    fig_idx=fig_idx+1
'''