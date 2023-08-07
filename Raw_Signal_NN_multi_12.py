# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 02:49:00 2020

@author: Povilas-Predator-PC
"""

from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
import h5py

import tensorflow as tf

import keras

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixEqualized_v2.mat')
# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixEqualized_53_multi_12_averaged_v2.mat')
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong05.mat')
# mat_contents = h5py.File(mat_fname,'r')
mat_contents = sio.loadmat(mat_fname)
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLong0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong15.mat')
# mat_contents = h5py.File(mat_fname,'r')
mat_contents = sio.loadmat(mat_fname)
DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLong1']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong25.mat')
# mat_contents = h5py.File(mat_fname,'r')
mat_contents = sio.loadmat(mat_fname)
DataMatrix2 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLong2']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong35.mat')
# mat_contents = h5py.File(mat_fname,'r')
mat_contents = sio.loadmat(mat_fname)
DataMatrix3 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLong3']),0,1)
# DataMatrix = DataMatrix[0,:]

multi = 5
VECL = 500

# Input = DataMatrix[:,2:2+multi*VECL]
# Input = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL],DataMatrix2[:,2:2+multi*VECL],DataMatrix3[:,2:2+multi*VECL]))
Input = np.hstack(( DataMatrix0[2:2+multi*VECL,:],DataMatrix1[2:2+multi*VECL,:],DataMatrix2[2:2+multi*VECL,:],DataMatrix3[2:2+multi*VECL,:]))

# Label = DataMatrix[:,0]
# Label = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0],DataMatrix2[:,0],DataMatrix3[:,0]))
Label = np.hstack(( DataMatrix0[0,:],DataMatrix1[0,:],DataMatrix2[0,:],DataMatrix3[0,:]))
DataMatrix0 =[]
DataMatrix1 =[]
DataMatrix2 =[]
DataMatrix3 =[]
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

mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV05.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix0 = np.array(mat_contents['EqualisedDataMatrixLongV0'])
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV15.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix1 = np.array(mat_contents['EqualisedDataMatrixLongV1'])
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV25.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix2 = np.array(mat_contents['EqualisedDataMatrixLongV2'])
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV35.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix3 = np.array(mat_contents['EqualisedDataMatrixLongV3'])
# DataMatrix = DataMatrix[0,:]

# multi = 12
# VECL = 500

# Input = DataMatrix[:,2:2+multi*VECL]
InputV = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL],DataMatrix2[:,2:2+multi*VECL],DataMatrix3[:,2:2+multi*VECL]))
# InputV = np.hstack(( DataMatrix0[2:2+multi*VECL,:],DataMatrix1[2:2+multi*VECL,:],DataMatrix2[2:2+multi*VECL,:],DataMatrix3[2:2+multi*VECL,:]))

# Label = DataMatrix[:,0]
LabelV = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0],DataMatrix2[:,0],DataMatrix3[:,0]))
# LabelV = np.hstack(( DataMatrix0[0,:],DataMatrix1[0,:],DataMatrix2[0,:],DataMatrix3[0,:]))
DataMatrix0 =[]
DataMatrix1 =[]
DataMatrix2 =[]
DataMatrix3 =[]


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
    elif LabelV[iii] ==1 : 
        LabelVC[iii,0] = 0
        LabelVC[iii,1] = 1
        LabelVC[iii,2] = 0
        LabelVC[iii,3] = 0
    elif LabelV[iii] ==2 : 
        LabelVC[iii,0] = 0
        LabelVC[iii,1] = 0
        LabelVC[iii,2] = 1
        LabelVC[iii,3] = 0
    elif LabelV[iii] ==3 : 
        LabelVC[iii,0] = 0
        LabelVC[iii,1] = 0
        LabelVC[iii,2] = 0
        LabelVC[iii,3] = 1


# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.LSTM(128,batch_input_shape = (None,53,1),return_sequences = 0),
#             tf.keras.layers.Dense(3,activation='sigmoid', name="final"),
#         ]
# )

Input = np.transpose(Input)
# InputV = np.transpose(InputV)

# Input = Input.reshape(-1,VECL*multi,1)
# LabelC = LabelC.reshape(-1,4,1)


# InputV = InputV.reshape(-1,VECL*multi,1)
# LabelVC = LabelVC.reshape(-1,4,1)


Input = Input.reshape(-1,multi,VECL)
LabelC = LabelC.reshape(-1,4,1)


InputV = InputV.reshape(-1,multi,VECL)
LabelVC = LabelVC.reshape(-1,4,1)

# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi*VECL,1]),

#             tf.keras.layers.Conv1D(500,multi ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"),
#             tf.keras.layers.Conv1D(128*2,5 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=2),
#             tf.keras.layers.Conv1D(20*2,1 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=2),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(128*2, activation ='relu'),
#             tf.keras.layers.Dense(128*2, activation ='relu'),
#             tf.keras.layers.Dense(64*2, activation ='relu'),
#             tf.keras.layers.Dense(4,activation ='softmax', name="final"),
#         ]
# )

# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi*VECL]),
            
#             tf.keras.layers.Dense(VECL, activation ='relu'),
#             tf.keras.layers.Dense(VECL, activation ='relu'),
#             tf.keras.layers.Dense(VECL, activation ='relu'),
#             tf.keras.layers.Dense(VECL, activation ='relu'),
 


#             tf.keras.layers.Dense(4,activation ='softmax', name="final"),
#         ]
# )


model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi,VECL]),
            
            tf.keras.layers.LSTM(VECL*multi),
            # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
            # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
            # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
            
            tf.keras.layers.Dense(4,activation ='softmax', name="final"),
        ]
)

# model = tf.keras.models.load_model('Feature_model_53_v2/1900')

# model = tf.keras.models.load_model('Feature_model_53_CNN_aveg_v1/200')


model.summary()

# model.compile('adam', loss='mse')
# model.compile('sgd', loss='binary_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
#return model.summary()
# history = model.fit(Input,LabelF,epochs = 20,)

for iii in range(1,10):
    history = model.fit(Input,LabelC,epochs = 10,batch_size =64,validation_data =(InputV,LabelVC))
    model.save(pjoin('RawSignalLong5_LSTM_v1/',str(iii*10+90)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
    

# plt.plot(np.argmax(model.predict(InputV),1))
# plt.plot(LabelV)








