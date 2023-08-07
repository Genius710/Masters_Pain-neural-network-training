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

multi = 1
VECL = 68



# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMean01Features_recog.mat')
# mat_contents = h5py.File(mat_fname,'r')
# DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMean0']),0,1)
# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMean11Features_recog.mat')
# mat_contents = h5py.File(mat_fname,'r')
# DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMean1']),0,1)

# InputMean1 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
# LabelMean1 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

# DataMatrix =[]     
        

# [x] = LabelMean1.shape
# LabelMean1C = np.empty([x,2], dtype=int)
       
# for iii in range(0,x):
    
#     if LabelMean1[iii] ==0:
#         LabelMean1C[iii,0] = 1
#         LabelMean1C[iii,1] = 0


#     elif LabelMean1[iii] ==1: 
#         LabelMean1C[iii,0] = 0
#         LabelMean1C[iii,1] = 1
        
        
# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMeanV01Features_recog.mat')
# mat_contents = h5py.File(mat_fname,'r')
# DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMeanV0']),0,1)
# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMeanV11Features_recog.mat')
# mat_contents = h5py.File(mat_fname,'r')
# DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMeanV1']),0,1)

# InputMeanV1 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
# LabelMeanV1 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

# DataMatrix =[]     
        

# [x] = LabelMeanV1.shape
# LabelMeanV1C = np.empty([x,2], dtype=int)
       
# for iii in range(0,x):
    
#     if LabelMeanV1[iii] ==0:
#         LabelMeanV1C[iii,0] = 1
#         LabelMeanV1C[iii,1] = 0


#     elif LabelMeanV1[iii] ==1: 
#         LabelMeanV1C[iii,0] = 0
#         LabelMeanV1C[iii,1] = 1
        
# InputMean1CNN = InputMean1.reshape(-1,VECL*multi,1)
# LabelMean1CCNN = LabelMean1C.reshape(-1,2,1)


# InputMeanV1CNN = InputMeanV1.reshape(-1,VECL*multi,1)
# LabelMeanV1CCNN = LabelMeanV1C.reshape(-1,2,1)

# InputMean1LSTM = InputMean1.reshape(-1,multi,VECL)
# LabelMean1CLSTM = LabelMean1C.reshape(-1,2,1)


# InputMeanV1LSTM = InputMeanV1.reshape(-1,multi,VECL)
# LabelMeanV1CLSTM = LabelMeanV1C.reshape(-1,2,1)        
        
        
        
multi =1       
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMean05Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMean0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMean15Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMean1']),0,1)

InputMean5 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
LabelMean5 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

DataMatrix =[]     
        

[x] = LabelMean5.shape
LabelMean5C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelMean5[iii] ==0:
        LabelMean5C[iii,0] = 1
        LabelMean5C[iii,1] = 0


    elif LabelMean5[iii] ==1: 
        LabelMean5C[iii,0] = 0
        LabelMean5C[iii,1] = 1
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMeanV05Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMeanV0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMeanV15Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMeanV1']),0,1)

InputMeanV5 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
LabelMeanV5 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

DataMatrix =[]     
        

[x] = LabelMeanV5.shape
LabelMeanV5C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanV5[iii] ==0:
        LabelMeanV5C[iii,0] = 1
        LabelMeanV5C[iii,1] = 0


    elif LabelMeanV5[iii] ==1: 
        LabelMeanV5C[iii,0] = 0
        LabelMeanV5C[iii,1] = 1
        
InputMean5CNN = InputMean5.reshape(-1,VECL*multi,1)
LabelMean5CCNN = LabelMean5C.reshape(-1,2,1)


InputMeanV5CNN = InputMeanV5.reshape(-1,VECL*multi,1)
LabelMeanV5CCNN = LabelMeanV5C.reshape(-1,2,1)

InputMean5LSTM = InputMean5.reshape(-1,multi,VECL)
LabelMean5CLSTM = LabelMean5C.reshape(-1,2,1)


InputMeanV5LSTM = InputMeanV5.reshape(-1,multi,VECL)
LabelMeanV5CLSTM = LabelMeanV5C.reshape(-1,2,1)                



multi =1     
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMean012Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMean0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMean112Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMean1']),0,1)

InputMean12 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
LabelMean12 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

DataMatrix =[]     
        

[x] = LabelMean12.shape
LabelMean12C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelMean12[iii] ==0:
        LabelMean12C[iii,0] = 1
        LabelMean12C[iii,1] = 0


    elif LabelMean12[iii] ==1: 
        LabelMean12C[iii,0] = 0
        LabelMean12C[iii,1] = 1
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMeanV012Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMeanV0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMeanV112Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixMeanV1']),0,1)

InputMeanV12 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
LabelMeanV12 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

DataMatrix =[]     
        

[x] = LabelMeanV12.shape
LabelMeanV12C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanV12[iii] ==0:
        LabelMeanV12C[iii,0] = 1
        LabelMeanV12C[iii,1] = 0


    elif LabelMeanV12[iii] ==1: 
        LabelMeanV12C[iii,0] = 0
        LabelMeanV12C[iii,1] = 1
        
InputMean12CNN = InputMean12.reshape(-1,VECL*multi,1)
LabelMean12CCNN = LabelMean12C.reshape(-1,2,1)


InputMeanV12CNN = InputMeanV12.reshape(-1,VECL*multi,1)
LabelMeanV12CCNN = LabelMeanV12C.reshape(-1,2,1)

InputMean12LSTM = InputMean12.reshape(-1,multi,VECL)
LabelMean12CLSTM = LabelMean12C.reshape(-1,2,1)


InputMeanV12LSTM = InputMeanV12.reshape(-1,multi,VECL)
LabelMeanV12CLSTM = LabelMeanV12C.reshape(-1,2,1)    





# multi =1
eep = 2
epoch_no = 1
size = 64;

    
multi = 1


    
    
    
model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi*VECL]),
            
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),

            tf.keras.layers.Dense(2,activation ='softmax', name="final"),
        ]
)

model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean5 ,LabelMean5C ,epochs = epoch_no,batch_size =size,validation_data =(InputMeanV5 ,LabelMeanV5C ))
    model.save(pjoin('RawSignalMean5Features_MLP_v1_recog/',str(iii*epoch_no+0)))
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
            tf.keras.layers.Dense(2,activation ='softmax', name="final"),
        ]
)
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean5CNN,LabelMean5CCNN,epochs = epoch_no,batch_size =size,validation_data =(InputMeanV5CNN,LabelMeanV5CCNN))
    model.save(pjoin('RawSignalMean5Features_CNN_v1_recog/',str(iii*epoch_no+100)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])

model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi,VECL]),
            
            tf.keras.layers.LSTM(VECL*multi),
            # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
            # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
            # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
            
            tf.keras.layers.Dense(2,activation ='softmax', name="final"),
        ]
)
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean5LSTM,LabelMean5CLSTM,epochs = epoch_no,batch_size =size,validation_data =(InputMeanV5LSTM,LabelMeanV5CLSTM))
    model.save(pjoin('RawSignalMean5Features_LSTM_v1_recog/',str(iii*epoch_no+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])



multi = 1
# eep = 2
# epoch_no = 1
    

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
            tf.keras.layers.Dense(2,activation ='softmax', name="final"),
        ]
)

model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean12CNN,LabelMean12CCNN,epochs = epoch_no,batch_size =size,validation_data =(InputMeanV12CNN,LabelMeanV12CCNN))
    model.save(pjoin('RawSignalMean12Features_CNN_v1_recog/',str(iii*epoch_no+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
    

model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi*VECL]),
            
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),

            tf.keras.layers.Dense(2,activation ='softmax', name="final"),
        ]
)

model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean12 ,LabelMean12C ,epochs = epoch_no,batch_size =size,validation_data =(InputMeanV12 ,LabelMeanV12C ))
    model.save(pjoin('RawSignalMean12Features_MLP_v1_recog/',str(iii*epoch_no+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])

model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi,VECL]),
            
            tf.keras.layers.LSTM(VECL*multi),
            # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
            # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
            # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
            
            tf.keras.layers.Dense(2,activation ='softmax', name="final"),
        ]
)

model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputMean12LSTM,LabelMean12CLSTM,epochs = epoch_no,batch_size =size,validation_data =(InputMeanV12LSTM,LabelMeanV12CLSTM))
    model.save(pjoin('RawSignalMean12Features_LSTM_v1_recog/',str(iii*epoch_no+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])





