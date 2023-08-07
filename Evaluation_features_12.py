# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 02:49:00 2020

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

# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixEqualized_v2.mat')
# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixEqualized_53_multi_12_averaged_v2.mat')
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixFeatures12.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=1)
DataMatrix = mat_contents['EqualisedDataMatrixFeatures']
# DataMatrix = DataMatrix[0,:]



multi =12
VACL = 68

Input = DataMatrix[:,2:2+multi*VACL]
Label = DataMatrix[:,0]

# Input = np.empty([1,53], dtype=float)
# Label = np.empty([1], dtype=int)

# for iii in range(0,50):
#     if (iii ==2 or iii ==20): 
#         print(iii)
#     else:
#         Input =  np.vstack(( Input,DataMatrix[iii][:,2:55]))
#         Label = np.concatenate(( Label,DataMatrix[iii][:,0]))
        
        

[x] = Label.shape
LabelC = np.empty([x,4], dtype=int)
       
for iii in range(0,x):
    
    if Label[iii] ==0:
        LabelC[iii,0] = 1
        LabelC[iii,1] = 0
        LabelC[iii,2] = 0
        LabelC[iii,3] = 0

    elif Label[iii] ==1: 
        LabelC[iii,0] = 0
        LabelC[iii,1] = 1
        LabelC[iii,2] = 0
        LabelC[iii,3] = 0


    elif Label[iii] ==2 : 
        LabelC[iii,0] = 0
        LabelC[iii,1] = 0
        LabelC[iii,2] = 1
        LabelC[iii,3] = 0
    elif Label[iii] ==3 : 
        LabelC[iii,0] = 0
        LabelC[iii,1] = 0
        LabelC[iii,2] = 0
        LabelC[iii,3] = 1

mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixFeaturesV12.mat')
# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixValidationSelected_multi_12_averaged_v2.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=1)
DataMatrix = mat_contents['DataMatrixFeaturesV']
# DataMatrix = DataMatrix[0,:]

InputV = DataMatrix[:,2:2+multi*VACL]
LabelV = DataMatrix[:,0]


# InputV = np.empty([1,53], dtype=float)
# LabelV = np.empty([1], dtype=int)

# for iii in range(50,51):
#     if (iii ==2 or iii ==20): 
#         print(iii)
#     else:
#         InputV =  np.vstack(( InputV,DataMatrix[iii][:,2:55]))
#         LabelV = np.concatenate(( LabelV,DataMatrix[iii][:,0]))
        
[x] = LabelV.shape
LabelVC = np.empty([x,4], dtype=int)
       
for iii in range(0,x):
    
    if LabelV[iii] ==0:
        LabelVC[iii,0] = 1
        LabelVC[iii,1] = 0
        LabelVC[iii,2] = 0
        LabelVC[iii,3] = 0
        LastZero = iii
    elif LabelV[iii] ==1 : 
        LabelVC[iii,0] = 0
        LabelVC[iii,1] = 1
        LabelVC[iii,2] = 0
        LabelVC[iii,3] = 0
        LastOne = iii
    elif LabelV[iii] ==2 : 
        LabelVC[iii,0] = 0
        LabelVC[iii,1] = 0
        LabelVC[iii,2] = 1
        LabelVC[iii,3] = 0
        LastTwo = iii
    elif LabelV[iii] ==3 : 
        LabelVC[iii,0] = 0
        LabelVC[iii,1] = 0
        LabelVC[iii,2] = 0
        LabelVC[iii,3] = 1
        LastThree = iii
        
        
        
# [x] = Label_IR.shape
# LabelC_IR = np.empty([x,4], dtype=int)
       
# for iii in range(0,x):
    
#     if Label_IR[iii] ==0:
#         LabelC_IR[iii,0] = 1
#         LabelC_IR[iii,1] = 0
#         LabelC_IR[iii,2] = 0
#         LabelC_IR[iii,3] = 0
#         LastZero = iii
#     elif Label_IR[iii] ==1 : 
#         LabelC_IR[iii,0] = 0
#         LabelC_IR[iii,1] = 1
#         LabelC_IR[iii,2] = 0
#         LabelC_IR[iii,3] = 0
#         LastOne = iii
#     elif Label_IR[iii] ==2 : 
#         LabelC_IR[iii,0] = 0
#         LabelC_IR[iii,1] = 0
#         LabelC_IR[iii,2] = 1
#         LabelC_IR[iii,3] = 0
#         LastTwo = iii
#     elif Label_IR[iii] ==3 : 
#         LabelC_IR[iii,0] = 0
#         LabelC_IR[iii,1] = 0
#         LabelC_IR[iii,2] = 0
#         LabelC_IR[iii,3] = 1
#         LastThree = iii
        
        
# CNN MLP
Input = Input.reshape(-1,VACL*multi,1)
LabelC = LabelC.reshape(-1,4,1)


InputV = InputV.reshape(-1,VACL*multi,1)
LabelVC = LabelVC.reshape(-1,4,1)

# LSTM
# Input = Input.reshape(-1,multi,VACL)
# LabelC = LabelC.reshape(-1,4,1)


# InputV = InputV.reshape(-1,multi,VACL)
# LabelVC = LabelVC.reshape(-1,4,1)



# model = tf.keras.models.load_model('Feature_model_68_MLP_multi_12_v1/900')
# model = tf.keras.models.load_model('Feature_model_68_LSTM_multi_12_v1/500')
model = tf.keras.models.load_model('Feature_model_68_CNN_multi_12_v1/300')


model.summary()

model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

BN0 = model.evaluate(InputV[0:LastZero],LabelVC[0:LastZero])
np.savetxt('Feature_model_68_CNN_multi_12_v1_results.txt',np.argmax(model.predict(InputV),1))
BN1 = model.evaluate(InputV[LastZero+1:LastOne],LabelVC[LastZero+1:LastOne])
BN2 = model.evaluate(InputV[LastOne+1:LastTwo],LabelVC[LastOne+1:LastTwo])
BN3 = model.evaluate(InputV[LastTwo+1:LastThree],LabelVC[LastTwo+1:LastThree])

# for iii in range(1,10):
#     history = model.fit(Input,LabelC,epochs = 100,batch_size =320*2,validation_data =(InputV,LabelVC))
#     model.save(pjoin('Feature_model_68_LSTM_multi_1_v1/',str(iii*100)))
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.plot(history.history['categorical_accuracy'])
    

plt.plot(np.argmax(model.predict(InputV),1))
plt.plot(LabelV)


# np.savetxt('Feature_model_68_MLP_multi_12_v1_results.txt',np.argmax(model.predict(InputV),1))
# np.savetxt('Feature_model_68_LSTM_multi_12_v1_results.txt',np.argmax(model.predict(InputV),1))

# f = open("Feature_model_68_CNN_multi_5_v1_results.txt", "w")
# f.write(np.argmax(model.predict(InputV),1))
# f.close()








