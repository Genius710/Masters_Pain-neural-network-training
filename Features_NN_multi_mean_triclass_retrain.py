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

# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixFeatures1Features_triclass.mat')
# mat_contents = sio.loadmat(mat_fname)
# DataMatrix = np.array(mat_contents['EqualisedDataMatrixFeatures'])

# multi = 1
VECL = 68


# InputMean1 = DataMatrix[:,2:2+multi*VECL]
# LabelMean1 = DataMatrix[:,0]

# DataMatrix =[]     
        

# [x] = LabelMean1.shape
# LabelMean1C = np.empty([x,3], dtype=int)
       
# for iii in range(0,x):
    
#     if LabelMean1[iii] ==0:
#         LabelMean1C[iii,0] = 1
#         LabelMean1C[iii,1] = 0
#         LabelMean1C[iii,2] = 0

#     elif LabelMean1[iii] ==1: 
#         LabelMean1C[iii,0] = 0
#         LabelMean1C[iii,1] = 1
#         LabelMean1C[iii,2] = 0


#     elif LabelMean1[iii] ==2 : 
#         LabelMean1C[iii,0] = 0
#         LabelMean1C[iii,1] = 0
#         LabelMean1C[iii,2] = 1

        
# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\\EqualisedDataMatrixFeaturesV1Features_triclass.mat')
# mat_contents = sio.loadmat(mat_fname)
# DataMatrix = np.array(mat_contents['EqualisedDataMatrixFeaturesV'])

# InputMean1V = DataMatrix[:,2:2+multi*VECL]
# LabelMean1V = DataMatrix[:,0]

# DataMatrix =[]     

    
# [x] = LabelMean1V.shape
# LabelMean1VC = np.empty([x,3], dtype=int)
       
# for iii in range(0,x):
    
#     if LabelMean1V[iii] ==0:
#         LabelMean1VC[iii,0] = 1
#         LabelMean1VC[iii,1] = 0
#         LabelMean1VC[iii,2] = 0
        
#     elif LabelMean1V[iii] ==1 : 
#         LabelMean1VC[iii,0] = 0
#         LabelMean1VC[iii,1] = 1
#         LabelMean1VC[iii,2] = 0

#     elif LabelMean1V[iii] ==2 : 
#         LabelMean1VC[iii,0] = 0
#         LabelMean1VC[iii,1] = 0
#         LabelMean1VC[iii,2] = 1


# InputMean1CNN = InputMean1.reshape(-1,VECL*multi,1)
# LabelMean1CCNN = LabelMean1C.reshape(-1,3,1)


# InputMean1VCNN = InputMean1V.reshape(-1,VECL*multi,1)
# LabelMean1VCCNN = LabelMean1VC.reshape(-1,3,1)

# InputMean1LSTM = InputMean1.reshape(-1,multi,VECL)
# LabelMean1CLSTM = LabelMean1C.reshape(-1,3,1)


# InputMean1VLSTM = InputMean1V.reshape(-1,multi,VECL)
# LabelMean1VCLSTM = LabelMean1VC.reshape(-1,3,1)


mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMean5Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixMean'])

multi = 1
VECL = 68


InputMean5 = DataMatrix[:,2:2+multi*VECL]
LabelMean5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMean5.shape
LabelMean5C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelMean5[iii] ==0:
        LabelMean5C[iii,0] = 1
        LabelMean5C[iii,1] = 0
        LabelMean5C[iii,2] = 0

    elif LabelMean5[iii] ==1: 
        LabelMean5C[iii,0] = 0
        LabelMean5C[iii,1] = 1
        LabelMean5C[iii,2] = 0


    elif LabelMean5[iii] ==2 : 
        LabelMean5C[iii,0] = 0
        LabelMean5C[iii,1] = 0
        LabelMean5C[iii,2] = 1

        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\\EqualisedDataMatrixMeanV5Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixMeanV'])

InputMean5V = DataMatrix[:,2:2+multi*VECL]
LabelMean5V = DataMatrix[:,0]

DataMatrix =[]     

    
[x] = LabelMean5V.shape
LabelMean5VC = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelMean5V[iii] ==0:
        LabelMean5VC[iii,0] = 1
        LabelMean5VC[iii,1] = 0
        LabelMean5VC[iii,2] = 0
        
    elif LabelMean5V[iii] ==1 : 
        LabelMean5VC[iii,0] = 0
        LabelMean5VC[iii,1] = 1
        LabelMean5VC[iii,2] = 0

    elif LabelMean5V[iii] ==2 : 
        LabelMean5VC[iii,0] = 0
        LabelMean5VC[iii,1] = 0
        LabelMean5VC[iii,2] = 1


InputMean5CNN = InputMean5.reshape(-1,VECL*multi,1)
LabelMean5CCNN = LabelMean5C.reshape(-1,3,1)


InputMean5VCNN = InputMean5V.reshape(-1,VECL*multi,1)
LabelMean5VCCNN = LabelMean5VC.reshape(-1,3,1)

InputMean5LSTM = InputMean5.reshape(-1,multi,VECL)
LabelMean5CLSTM = LabelMean5C.reshape(-1,3,1)


InputMean5VLSTM = InputMean5V.reshape(-1,multi,VECL)
LabelMean5VCLSTM = LabelMean5VC.reshape(-1,3,1)

mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMean12Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixMean'])

multi = 1
VECL = 68


InputMean12 = DataMatrix[:,2:2+multi*VECL]
LabelMean12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMean12.shape
LabelMean12C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelMean12[iii] ==0:
        LabelMean12C[iii,0] = 1
        LabelMean12C[iii,1] = 0
        LabelMean12C[iii,2] = 0

    elif LabelMean12[iii] ==1: 
        LabelMean12C[iii,0] = 0
        LabelMean12C[iii,1] = 1
        LabelMean12C[iii,2] = 0


    elif LabelMean12[iii] ==2 : 
        LabelMean12C[iii,0] = 0
        LabelMean12C[iii,1] = 0
        LabelMean12C[iii,2] = 1

        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\\EqualisedDataMatrixMeanV12Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixMeanV'])

InputMean12V = DataMatrix[:,2:2+multi*VECL]
LabelMean12V = DataMatrix[:,0]

DataMatrix =[]     

    
[x] = LabelMean12V.shape
LabelMean12VC = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelMean12V[iii] ==0:
        LabelMean12VC[iii,0] = 1
        LabelMean12VC[iii,1] = 0
        LabelMean12VC[iii,2] = 0
        
    elif LabelMean12V[iii] ==1 : 
        LabelMean12VC[iii,0] = 0
        LabelMean12VC[iii,1] = 1
        LabelMean12VC[iii,2] = 0

    elif LabelMean12V[iii] ==2 : 
        LabelMean12VC[iii,0] = 0
        LabelMean12VC[iii,1] = 0
        LabelMean12VC[iii,2] = 1


InputMean12CNN = InputMean12.reshape(-1,VECL*multi,1)
LabelMean12CCNN = LabelMean12C.reshape(-1,3,1)


InputMean12VCNN = InputMean12V.reshape(-1,VECL*multi,1)
LabelMean12VCCNN = LabelMean12VC.reshape(-1,3,1)

InputMean12LSTM = InputMean12.reshape(-1,multi,VECL)
LabelMean12CLSTM = LabelMean12C.reshape(-1,3,1)


InputMean12VLSTM = InputMean12V.reshape(-1,multi,VECL)
LabelMean12VCLSTM = LabelMean12VC.reshape(-1,3,1)

eep = 11
epoch_no = 50
size = 256



# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi*VECL,1]),

#             tf.keras.layers.Conv1D(64,multi ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"),
#             tf.keras.layers.Conv1D(128*2,5 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=2),
#             tf.keras.layers.Conv1D(20*2,1 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=2),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(128*2, activation ='relu'),
#             tf.keras.layers.Dense(128*2, activation ='relu'),
#             tf.keras.layers.Dense(64*2, activation ='relu'),
#             tf.keras.layers.Dense(2,activation ='softmax', name="final"),
#         ]
# )

# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi*VECL]),
            
#             tf.keras.layers.Dense(VECL, activation ='relu'),
#             tf.keras.layers.Dense(VECL, activation ='relu'),
#             tf.keras.layers.Dense(VECL, activation ='relu'),
#             tf.keras.layers.Dense(VECL, activation ='relu'),
 


#             tf.keras.layers.Dense(2,activation ='softmax', name="final"),
#         ]
# )


# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi,VECL]),
            
#             tf.keras.layers.LSTM(VECL*multi),
#             tf.keras.layers.Dense(2,activation ='softmax', name="final"),
#         ]
# )


# model = tf.keras.models.load_model('RawSignalMean1Features_MLP_v1_triclass/1')
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

# for iii in range(1,eep):
#     history = model.fit(InputMean1 ,LabelMean1C ,epochs = epoch_no,batch_size =size,validation_data =(InputMean1V ,LabelMean1VC ))
#     model.save(pjoin('RawSignalMean1Features_MLP_v1_triclass/',str(iii*epoch_no+0)))
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.plot(history.history['categorical_accuracy'])
    
model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi*VECL]),
            
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
 


            tf.keras.layers.Dense(3,activation ='softmax', name="final"),
        ]
)
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean12 ,LabelMean12C ,epochs = epoch_no,batch_size =size,validation_data =(InputMean12V ,LabelMean12VC ))
    model.save(pjoin('RawSignalMean12Features_MLP_v1_triclass/',str(iii*epoch_no+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
    
model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi*VECL]),
            
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
 


            tf.keras.layers.Dense(3,activation ='softmax', name="final"),
        ]
)
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean5 ,LabelMean5C ,epochs = epoch_no,batch_size =size,validation_data =(InputMean5V ,LabelMean5VC ))
    model.save(pjoin('RawSignalMean5Features_MLP_v1_triclass/',str(iii*epoch_no+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])


# model = tf.keras.models.load_model('RawSignalMean1Features_LSTM_v1_triclass/1')
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

# for iii in range(1,eep):
#     history = model.fit(InputMean1LSTM,LabelMean1CLSTM,epochs = epoch_no,batch_size =size,validation_data =(InputMean1VLSTM,LabelMean1VCLSTM))
#     model.save(pjoin('RawSignalMean1Features_LSTM_v1_triclass/',str(iii*epoch_no+0)))
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.plot(history.history['categorical_accuracy'])
    
model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi,VECL]),
            
            tf.keras.layers.LSTM(VECL*multi),
            tf.keras.layers.Dense(3,activation ='softmax', name="final"),
        ]
)
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean12LSTM,LabelMean12CLSTM,epochs = epoch_no,batch_size =size,validation_data =(InputMean12VLSTM,LabelMean12VCLSTM))
    model.save(pjoin('RawSignalMean12Features_LSTM_v1_triclass/',str(iii*epoch_no+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
    
model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi,VECL]),
            
            tf.keras.layers.LSTM(VECL*multi),
            tf.keras.layers.Dense(3,activation ='softmax', name="final"),
        ]
)
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean5LSTM,LabelMean5CLSTM,epochs = epoch_no,batch_size =size,validation_data =(InputMean5VLSTM,LabelMean5VCLSTM))
    model.save(pjoin('RawSignalMean5Features_LSTM_v1_triclass/',str(iii*epoch_no+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])


# model = tf.keras.models.load_model('RawSignalMean1Features_CNN_v1_triclass/1')
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

# for iii in range(1,eep):
#     history = model.fit(InputMean1CNN,LabelMean1CCNN,epochs = epoch_no,batch_size =size,validation_data =(InputMean1VCNN,LabelMean1VCCNN))
#     model.save(pjoin('RawSignalMean1Features_CNN_v1_triclass/',str(iii*epoch_no+0)))
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.plot(history.history['categorical_accuracy'])
    
model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi*VECL,1]),

            tf.keras.layers.Conv1D(64,multi ,activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"),
            tf.keras.layers.Conv1D(128*2,5 ,activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(20*2,1 ,activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128*2, activation ='relu'),
            tf.keras.layers.Dense(128*2, activation ='relu'),
            tf.keras.layers.Dense(64*2, activation ='relu'),
            tf.keras.layers.Dense(3,activation ='softmax', name="final"),
        ]
)
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean12CNN,LabelMean12CCNN,epochs = epoch_no,batch_size =size,validation_data =(InputMean12VCNN,LabelMean12VCCNN))
    model.save(pjoin('RawSignalMean12Features_CNN_v1_triclass/',str(iii*epoch_no+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
    
model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi*VECL,1]),

            tf.keras.layers.Conv1D(64,multi ,activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"),
            tf.keras.layers.Conv1D(128*2,5 ,activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(20*2,1 ,activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128*2, activation ='relu'),
            tf.keras.layers.Dense(128*2, activation ='relu'),
            tf.keras.layers.Dense(64*2, activation ='relu'),
            tf.keras.layers.Dense(3,activation ='softmax', name="final"),
        ]
)
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean5CNN,LabelMean5CCNN,epochs = epoch_no,batch_size =size,validation_data =(InputMean5VCNN,LabelMean5VCCNN))
    model.save(pjoin('RawSignalMean5Features_CNN_v1_triclass/',str(iii*epoch_no+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])









