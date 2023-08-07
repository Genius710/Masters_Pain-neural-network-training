# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:34:50 2020

@author: Povilas-Predator-PC
"""

from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np

import tensorflow as tf

import keras

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


model = tf.keras.models.load_model('LSTMv2_epoch\99')

# blacklist = [22,24,35,38]
# whitelist = [25,27,28,29,41,46,48]
# soso = [23,26,30,31,32,33,34,36,37,40,43,44,45,49,50,51]

# whitelist = [23,25,26,27,28,29,30,31,32,34,35,36,37,38,40,41,43,44,45,46,48,49,50,51]

num = str(1)

mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'CNAPBlock.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
InputN = mat_contents['Input']
LabelN = mat_contents['Label']
NPRSN = mat_contents['NPRS']

whitelist = [23,25,26,27,28,29,30,31,32,34,35,36,37,38,40,41,43,44,45,46,48,49,50,51]
size = len(whitelist)+21
for iii in range(2,22):
    num = str(iii)
    if iii <10:
        print('Aurimod0' +num +'CNAPBlock.mat')
        mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'CNAPBlock.mat')
    else:
        print('Aurimod' +num +'CNAPBlock.mat')
        mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod' +num +'CNAPBlock.mat') 
    mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
    Input = mat_contents['Input']
    Label = mat_contents['Label']
    NPRS = mat_contents['NPRS']
    InputN = np.vstack((InputN,Input))
    LabelN = np.vstack((LabelN,Label))
    NPRSN = np.hstack((NPRSN,NPRS))


for iii in range(21,size):
    num = str(whitelist[iii-21])
    # num = str(whitelist)
    if iii <10:
        print('Aurimod0' +num +'CNAPBlock.mat')
        mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'CNAPBlock.mat')
    else:
        print('Aurimod' +num +'CNAPBlock.mat')
        mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod' +num +'CNAPBlock.mat') 
    mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
    Input = mat_contents['Input']
    Label = mat_contents['Label']
    NPRS = mat_contents['NPRS']
    InputN = np.vstack((InputN,Input))
    LabelN = np.vstack((LabelN,Label))
    NPRSN = np.hstack((NPRSN,NPRS))

InputN = InputN.reshape(-1,104,1)

x = max(InputN.shape)
result = np.empty([x,5], dtype=float)
resultN = np.empty([x], dtype=float)

result = model.predict(InputN)
    
    
for iii in range(0,x):
    resultN[iii] = np.argmax(result[iii,:])
    
plt.figure(dpi=300)
plt.plot(resultN)  
# plt.plot(NPRSN)
plt.ylim(0,5)

[x,y] = NPRSN.shape
NPRSNF = np.empty([y,5], dtype=int)
var = np.empty(1, dtype=int)
NPRSN =NPRSN[0]
for iii in range(0,y):
    # var = np.round(NPRSN[iii]/10)
    # var = var.astype(int)
    # if var ==0:
    #     NPRSNF[iii,:] =np.hstack((np.ones(1),np.zeros(9)))
    # else:
    #     NPRSNF[iii,:] =np.hstack(
    #         (
    #             np.zeros(var-1),
    #             np.ones(1),
    #             np.zeros(10-var)
                
    #             )
    #         )
    if NPRSN[iii] ==0:
        NPRSNF[iii,0] = 1
        NPRSNF[iii,1] = 0
        NPRSNF[iii,2] = 0
        NPRSNF[iii,3] = 0
        NPRSNF[iii,4] = 0
    elif NPRSN[iii] > 0 and NPRSN[iii] <25: 
        NPRSNF[iii,0] = 0
        NPRSNF[iii,1] = 1
        NPRSNF[iii,2] = 0
        NPRSNF[iii,3] = 0
        NPRSNF[iii,4] = 0
    elif NPRSN[iii] >= 25 and NPRSN[iii] <50:
        NPRSNF[iii,0] = 0
        NPRSNF[iii,1] = 0
        NPRSNF[iii,2] = 1
        NPRSNF[iii,3] = 0
        NPRSNF[iii,4] = 0
    elif NPRSN[iii] >= 50 and NPRSN[iii] <75:
        NPRSNF[iii,0] = 0
        NPRSNF[iii,1] = 0
        NPRSNF[iii,2] = 0
        NPRSNF[iii,3] = 1
        NPRSNF[iii,4] = 0
    elif NPRSN[iii] >= 75:
        NPRSNF[iii,0] = 0
        NPRSNF[iii,1] = 0
        NPRSNF[iii,2] = 0
        NPRSNF[iii,3] = 0
        NPRSNF[iii,4] = 1

test_dataset = tf.data.Dataset.from_tensor_slices((InputN, NPRSNF))
test_dataset = test_dataset.batch(64)
print("Evaluate")

res = model.evaluate(test_dataset)
