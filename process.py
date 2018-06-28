#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 20:47:45 2018

@author: paco
"""
import pandas as pd
from package.training import BayesianGaussianMixtureRegressor as BGmr
from pandas import np

#
def compare_models(obj1, obj2):
    n1 = obj1.n_components
    n2 = obj2.n_components
    kernelParameters = {}
    equivalent_kernels = {}
    for kernel in range(n1):
        kernelParameters[kernel] = {}
        sigma1 = obj1.covariances_[kernel,:,:]
        mu1 = obj1.means_[kernel,:]
        tmp = {'kl':1e9, 'idx':-1}
        for kernel_tmp in range(n2):
            if kernel_tmp not in equivalent_kernels.values():
                mu2 = obj2.means_[kernel_tmp,:]
                sigma2 = obj2.covariances_[kernel_tmp,:,:]
                kl = np.abs(kullback_leibler_divergence(mu1, sigma1,
                                                        mu2, sigma2))
                if tmp['kl'] > kl:
                    tmp['kl'] = kl
                    tmp['idx'] = kernel_tmp
                    equivalent_kernels[kernel] = kernel_tmp
        
    return

def kullback_leibler_divergence(mu1,sigma1,mu2, sigma2):
    term1 = np.linalg.slogdet(sigma2)[1] - np.linalg.slogdet(sigma1)[1]
    sigma_inv =np.linalg.inv(sigma2)
    term2 = np.matrix.trace(np.dot(sigma_inv, sigma1))
    tmp = mu2 - mu1
    term3= np.dot(np.transpose(tmp), np.dot(sigma_inv,tmp))
    result = 0.5 * (term1 - tmp.shape[0] + term2 + term3)
    return result

def get_labels(df):
    labels = []
    for x in df['label'].tolist():
        if x not in labels:
            labels.append(x)
    return labels
def sin_cos(val):
    _x = pd.np.sin(2 * pd.np.pi * (pd.np.float(val) / 7.0))*10
    _y = pd.np.cos(2 * pd.np.pi * (pd.np.float(val) / 7.0))*10
    _x1 = pd.np.sin(2 * pd.np.pi * (pd.np.float(val) / 7.0 + 1/7.0))*10
    _y1 = pd.np.cos(2 * pd.np.pi * (pd.np.float(val) / 7.0 + 1/7.0))*10
    _x2 = pd.np.sin(2 * pd.np.pi * (pd.np.float(val) / 7.0 + 5/15.0))*10
    _y2 = pd.np.cos(2 * pd.np.pi * (pd.np.float(val) / 7.0 + 5/15.0))*10
    return _x, _y, _x1, _y1, _x2, _y2
# 
df_training = pd.read_csv('training_baseline.csv')
df_training.rename(columns={'224':'label'}, inplace=True)
training_labels = get_labels(df_training)
df_validation = pd.read_csv('test_baseline.csv')
df_validation.rename(columns={'224':'label'}, inplace=True)
validation_labels = get_labels(df_validation)
#
_mask_train = {}
_mask_validation = {}
df_training['x'] = 0
df_training['y'] = 0
df_training['x1'] = 0
df_training['y1'] = 0
df_training['x2'] = 0
df_training['y2'] = 0
df_validation['x'] = 0
df_validation['y'] = 0
df_validation['x1'] = 0
df_validation['y1'] = 0
df_validation['x2'] = 0
df_validation['y2'] = 0
for _c, _label in enumerate(training_labels):
    _mask_validation[_c] = (df_validation['label'] == _label)
    x, y, x1, y1, x2, y2 = sin_cos(_c)
    df_validation.loc[_mask_validation[_c]==True,['x']], df_validation.loc[_mask_validation[_c]==True,['y']] = x, y
    df_validation.loc[_mask_validation[_c]==True,['x1']], df_validation.loc[_mask_validation[_c]==True,['y1']] = x1, y1
    df_validation.loc[_mask_validation[_c]==True,['x2']], df_validation.loc[_mask_validation[_c]==True,['y2']] = x2, y2
    _mask_train[_c] = (df_training['label'] == _label)
    df_training.loc[_mask_train[_c]==True,['x']], df_training.loc[_mask_train[_c]==True,['y']] = x, y
    df_training.loc[_mask_train[_c]==True,['x1']], df_training.loc[_mask_train[_c]==True,['y1']] = x1, y1
    df_training.loc[_mask_train[_c]==True,['x2']], df_training.loc[_mask_train[_c]==True,['y2']] = x2, y2

# %%
lst = ['x', 'x1','x2','y','y1','y2']
n_rep = 50
results = {}
#n = {0:3, 1:3, 2:25, 3:19, 4:20, 5:5, 6:20}
n = {0: 3, 1: 3, 5: 3, 2: 7, 3: 4, 4: 4, 6: 8}
for _key in _mask_train.keys():
    results[_key] = {}
    for _r in range(n_rep):
        obj = BGmr(n_components=n[_key])
        obj.fit_pandas(df_training[_mask_train[_key]].drop(columns=['label']),
                       scaler=True, no_scale=lst)
        _w = obj.weights_.reshape(-1,1)
        if _r == 0:
            results[_key]['w'] = _w
            results[_key]['obj'] = [obj]
        else:
            results[_key]['w'] = pd.np.concatenate((results[_key]['w'], _w), axis=1)
            results[_key]['obj'].append(obj)
    
# %%

sigma1 = results[0]['obj'][0].covariances_[2,:,:]
sigma2 = results[0]['obj'][1].covariances_[2,:,:]
mu1 = results[0]['obj'][0].means_[2,:]
mu2 = results[0]['obj'][1].means_[2,:]
#%%
obj_y = BGmr(n_components=80)
obj_x = BGmr(n_components=80, max_iter=1000)

obj_x.fit_pandas(df_training.drop(columns=['label']))

#obj_y.fit_pandas(df_training.drop(columns=['label','x']))
#%%
w, df = obj_x.get_posterior_stats(df_validation.drop(columns=['label','x','y','x1','y1','x2','y2']))

#df = pd.concat((df_x,df_y),axis=1)
#df_comp = pd.concat((df.rename(columns={'x': 'x_s', 'y': 'y_s','z': 'z_s', 'v': 'v_s'}),
#                     df_validation[['x','y','z','v']]),
#                    axis=1)
import matplotlib.pyplot as plt
#%%
styles1 = ['bx','rx','yx','kx','gx','ro','ko']

for _key in _mask_validation.keys():
    if _key == 0:
        ax = df[_mask_validation[_key]==True].plot(x='x',y='y',style=[styles1[_key]],title='2')
    else:
        df[_mask_validation[_key]==True].plot(ax=ax,x='x',y='y',style=[styles1[_key]])
#%%

styles1 = ['bx','rx','yx','kx','gx','ro','ko']

for _key in _mask_validation.keys():
    if _key == 0:
        ax = df_validation[_mask_validation[_key]==True].plot(x='x',y='y',style=[styles1[_key]])
    else:
        df_validation[_mask_validation[_key]==True].plot(ax=ax,x='x',y='y',style=[styles1[_key]])
#%%
