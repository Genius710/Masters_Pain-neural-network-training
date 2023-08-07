# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:01:40 2020

@author: Povilas-Predator-PC
"""
from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np

import tensorflow as tf

import keras
model = tf.keras.models.load_model('seluv1')

num = str(1)

mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'PPGBlockC.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
InputN = mat_contents['Input']
LabelN = mat_contents['Label']
NPRSN = mat_contents['NPRS']

whitelist = [23,25,26,27,28,29,30,31,32,34,35,36,37,38,40,41,43,44,45,46,48,49,50,51]
print(len(whitelist))
size = len(whitelist)+22
for iii in range(2,52):
    num = str(iii)
    if iii <10:
        print('Aurimod0' +num +'CNAPBlock.mat')
        mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'PPGBlockC.mat')
    else:
        print('Aurimod' +num +'CNAPBlock.mat')
        mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod' +num +'PPGBlockC.mat') 
    mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
    Input = mat_contents['Input']
    Label = mat_contents['Label']
    NPRS = mat_contents['NPRS']
    InputN = np.vstack((InputN,Input))
    LabelN = np.vstack((LabelN,Label))
    NPRSN = np.hstack((NPRSN,NPRS))


# for iii in range(22,size):
#     print(iii)
#     num = str(whitelist[iii-22])
#     # num = str(whitelist)
#     if iii <10:
#         print('Aurimod0' +num +'CNAPBlock.mat')
#         mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'PPGBlockC.mat')
#     else:
#         print('Aurimod' +num +'CNAPBlock.mat')
#         mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod' +num +'PPGBlockC.mat') 
#     mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
#     Input = mat_contents['Input']
#     Label = mat_contents['Label']
#     NPRS = mat_contents['NPRS']
#     InputN = np.vstack((InputN,Input))
#     LabelN = np.vstack((LabelN,Label))
#     NPRSN = np.hstack((NPRSN,NPRS))
    


[x,y] = NPRSN.shape
NPRSNF = np.empty([y,5], dtype=float)
NPRSN =NPRSN[0]
for iii in range(0,y):
    
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
    
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# InputN = InputN.reshape(-1,104,1)
# LabelN = LabelN.reshape(-1,101,1)

# NPRSNF = NPRSNF.reshape(-1,5,1)

x_train,  y_train,x_test, y_test = train_test_split(InputN, LabelN, test_size=0.2)

# # physical_devices = tf.config.list_physical_devices('GPU') 
# # tf.config.experimental.set_memory_growth(physical_devices[0], True)

opt = tf.keras.optimizers.SGD(
    learning_rate=0.00001, momentum=0.0, nesterov=False, name="SGD")


model.compile(opt, loss='mean_squared_error')
model.fit(x_train,x_test,epochs = 2000,validation_data =(y_train,y_test))




x = max(InputN.shape)
result = np.empty([x,5], dtype=float)
resultN = np.empty([x], dtype=float)
result = model.predict(InputN)
    
    
for iii in range(0,x):
    resultN[iii] = np.argmax(result[iii,:])
    
plt.figure(dpi=600)
plt.plot(resultN)  
plt.plot(NPRSN)




















