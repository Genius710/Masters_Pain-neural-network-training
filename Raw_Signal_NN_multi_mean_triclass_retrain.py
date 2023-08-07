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

mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong1_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLong'])

multi = 1
VECL = 500


InputLong1 = DataMatrix[:,2:2+multi*VECL]
LabelLong1 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLong1.shape
LabelLong1C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLong1[iii] ==0:
        LabelLong1C[iii,0] = 1
        LabelLong1C[iii,1] = 0
        LabelLong1C[iii,2] = 0

    elif LabelLong1[iii] ==1: 
        LabelLong1C[iii,0] = 0
        LabelLong1C[iii,1] = 1
        LabelLong1C[iii,2] = 0


    elif LabelLong1[iii] ==2 : 
        LabelLong1C[iii,0] = 0
        LabelLong1C[iii,1] = 0
        LabelLong1C[iii,2] = 1

        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\\EqualisedDataMatrixLongV1_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLongV'])

InputLong1V = DataMatrix[:,2:2+multi*VECL]
LabelLong1V = DataMatrix[:,0]

DataMatrix =[]     

    
[x] = LabelLong1V.shape
LabelLong1VC = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLong1V[iii] ==0:
        LabelLong1VC[iii,0] = 1
        LabelLong1VC[iii,1] = 0
        LabelLong1VC[iii,2] = 0
        
    elif LabelLong1V[iii] ==1 : 
        LabelLong1VC[iii,0] = 0
        LabelLong1VC[iii,1] = 1
        LabelLong1VC[iii,2] = 0

    elif LabelLong1V[iii] ==2 : 
        LabelLong1VC[iii,0] = 0
        LabelLong1VC[iii,1] = 0
        LabelLong1VC[iii,2] = 1


InputLong1CNN = InputLong1.reshape(-1,VECL*multi,1)
LabelLong1CCNN = LabelLong1C.reshape(-1,3,1)


InputLong1VCNN = InputLong1V.reshape(-1,VECL*multi,1)
LabelLong1VCCNN = LabelLong1VC.reshape(-1,3,1)

InputLong1LSTM = InputLong1.reshape(-1,multi,VECL)
LabelLong1CLSTM = LabelLong1C.reshape(-1,3,1)


InputLong1VLSTM = InputLong1V.reshape(-1,multi,VECL)
LabelLong1VCLSTM = LabelLong1VC.reshape(-1,3,1)


mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong5_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLong'])

multi = 5
VECL = 500


InputLong5 = DataMatrix[:,2:2+multi*VECL]
LabelLong5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLong5.shape
LabelLong5C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLong5[iii] ==0:
        LabelLong5C[iii,0] = 1
        LabelLong5C[iii,1] = 0
        LabelLong5C[iii,2] = 0

    elif LabelLong5[iii] ==1: 
        LabelLong5C[iii,0] = 0
        LabelLong5C[iii,1] = 1
        LabelLong5C[iii,2] = 0


    elif LabelLong5[iii] ==2 : 
        LabelLong5C[iii,0] = 0
        LabelLong5C[iii,1] = 0
        LabelLong5C[iii,2] = 1

        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\\EqualisedDataMatrixLongV5_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLongV'])

InputLong5V = DataMatrix[:,2:2+multi*VECL]
LabelLong5V = DataMatrix[:,0]

DataMatrix =[]     

    
[x] = LabelLong5V.shape
LabelLong5VC = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLong5V[iii] ==0:
        LabelLong5VC[iii,0] = 1
        LabelLong5VC[iii,1] = 0
        LabelLong5VC[iii,2] = 0
        
    elif LabelLong5V[iii] ==1 : 
        LabelLong5VC[iii,0] = 0
        LabelLong5VC[iii,1] = 1
        LabelLong5VC[iii,2] = 0

    elif LabelLong5V[iii] ==2 : 
        LabelLong5VC[iii,0] = 0
        LabelLong5VC[iii,1] = 0
        LabelLong5VC[iii,2] = 1


InputLong5CNN = InputLong5.reshape(-1,VECL*multi,1)
LabelLong5CCNN = LabelLong5C.reshape(-1,3,1)


InputLong5VCNN = InputLong5V.reshape(-1,VECL*multi,1)
LabelLong5VCCNN = LabelLong5VC.reshape(-1,3,1)

InputLong5LSTM = InputLong5.reshape(-1,multi,VECL)
LabelLong5CLSTM = LabelLong5C.reshape(-1,3,1)


InputLong5VLSTM = InputLong5V.reshape(-1,multi,VECL)
LabelLong5VCLSTM = LabelLong5VC.reshape(-1,3,1)

mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong12_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLong'])

multi = 12
VECL = 500


InputLong12 = DataMatrix[:,2:2+multi*VECL]
LabelLong12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLong12.shape
LabelLong12C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLong12[iii] ==0:
        LabelLong12C[iii,0] = 1
        LabelLong12C[iii,1] = 0
        LabelLong12C[iii,2] = 0

    elif LabelLong12[iii] ==1: 
        LabelLong12C[iii,0] = 0
        LabelLong12C[iii,1] = 1
        LabelLong12C[iii,2] = 0


    elif LabelLong12[iii] ==2 : 
        LabelLong12C[iii,0] = 0
        LabelLong12C[iii,1] = 0
        LabelLong12C[iii,2] = 1

        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\\EqualisedDataMatrixLongV12_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLongV'])

InputLong12V = DataMatrix[:,2:2+multi*VECL]
LabelLong12V = DataMatrix[:,0]

DataMatrix =[]     

    
[x] = LabelLong12V.shape
LabelLong12VC = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLong12V[iii] ==0:
        LabelLong12VC[iii,0] = 1
        LabelLong12VC[iii,1] = 0
        LabelLong12VC[iii,2] = 0
        
    elif LabelLong12V[iii] ==1 : 
        LabelLong12VC[iii,0] = 0
        LabelLong12VC[iii,1] = 1
        LabelLong12VC[iii,2] = 0

    elif LabelLong12V[iii] ==2 : 
        LabelLong12VC[iii,0] = 0
        LabelLong12VC[iii,1] = 0
        LabelLong12VC[iii,2] = 1


InputLong12CNN = InputLong12.reshape(-1,VECL*multi,1)
LabelLong12CCNN = LabelLong12C.reshape(-1,3,1)


InputLong12VCNN = InputLong12V.reshape(-1,VECL*multi,1)
LabelLong12VCCNN = LabelLong12VC.reshape(-1,3,1)

InputLong12LSTM = InputLong12.reshape(-1,multi,VECL)
LabelLong12CLSTM = LabelLong12C.reshape(-1,3,1)


InputLong12VLSTM = InputLong12V.reshape(-1,multi,VECL)
LabelLong12VCLSTM = LabelLong12VC.reshape(-1,3,1)



# model = tf.keras.models.load_model('RawSignalMean1_MLP_v1_triclass/400')
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

# for iii in range(1,3):
#     history = model.fit(InputLong1 ,LabelLong1C ,epochs = 50,batch_size =32,validation_data =(InputLong1V ,LabelLong1VC ))
#     model.save(pjoin('RawSignalMean1_MLP_v1_triclass/',str(iii*50+400)))
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.plot(history.history['categorical_accuracy'])
    
# model = tf.keras.models.load_model('RawSignalLong12_MLP_v1_triclass/400')
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

# for iii in range(1,3):
#     history = model.fit(InputLong12 ,LabelLong12C ,epochs = 50,batch_size =32,validation_data =(InputLong12V ,LabelLong12VC ))
#     model.save(pjoin('RawSignalLong12_MLP_v1_triclass/',str(iii*50+400)))
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.plot(history.history['categorical_accuracy'])
    
# model = tf.keras.models.load_model('RawSignalLong5_MLP_v1_triclass/400')
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

# for iii in range(1,3):
#     history = model.fit(InputLong5 ,LabelLong5C ,epochs = 50,batch_size =32,validation_data =(InputLong5V ,LabelLong5VC ))
#     model.save(pjoin('RawSignalLong5_MLP_v1_triclass/',str(iii*50+400)))
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.plot(history.history['categorical_accuracy'])


# model = tf.keras.models.load_model('RawSignalMean1_LSTM_v1_triclass/400')
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

# for iii in range(1,3):
#     history = model.fit(InputLong1LSTM,LabelLong1CLSTM,epochs = 50,batch_size =32,validation_data =(InputLong1VLSTM,LabelLong1VCLSTM))
#     model.save(pjoin('RawSignalMean1_LSTM_v1_triclass/',str(iii*50+400)))
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.plot(history.history['categorical_accuracy'])
    
# model = tf.keras.models.load_model('RawSignalLong12_LSTM_v1_triclass/400')
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

# for iii in range(1,3):
#     history = model.fit(InputLong12LSTM,LabelLong12CLSTM,epochs = 50,batch_size =32,validation_data =(InputLong12VLSTM,LabelLong12VCLSTM))
#     model.save(pjoin('RawSignalLong12_LSTM_v1_triclass/',str(iii*50+400)))
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.plot(history.history['categorical_accuracy'])
    
# model = tf.keras.models.load_model('RawSignalLong5_LSTM_v1_triclass/400')
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

# for iii in range(1,3):
#     history = model.fit(InputLong5LSTM,LabelLong5CLSTM,epochs = 50,batch_size =32,validation_data =(InputLong5VLSTM,LabelLong5VCLSTM))
#     model.save(pjoin('RawSignalLong5_LSTM_v1_triclass/',str(iii*50+400)))
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.plot(history.history['categorical_accuracy'])


# model = tf.keras.models.load_model('RawSignalMean1_CNN_v1_triclass/400')
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

# for iii in range(1,3):
#     history = model.fit(InputLong1CNN,LabelLong1CCNN,epochs = 50,batch_size =32,validation_data =(InputLong1VCNN,LabelLong1VCCNN))
#     model.save(pjoin('RawSignalMean1_CNN_v1_triclass/',str(iii*50+400)))
#     plt.plot(history.history['val_categorical_accuracy'])
#     plt.plot(history.history['categorical_accuracy'])
    
model = tf.keras.models.load_model('RawSignalLong12_CNN_v1_triclass/400')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,3):
    history = model.fit(InputLong12CNN,LabelLong12CCNN,epochs = 50,batch_size =32,validation_data =(InputLong12VCNN,LabelLong12VCCNN))
    model.save(pjoin('RawSignalLong12_CNN_v1_triclass/',str(iii*50+400)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
    
model = tf.keras.models.load_model('RawSignalLong5_CNN_v1_triclass/400')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,3):
    history = model.fit(InputLong5CNN,LabelLong5CCNN,epochs = 50,batch_size =32,validation_data =(InputLong5VCNN,LabelLong5VCCNN))
    model.save(pjoin('RawSignalLong5_CNN_v1_triclass/',str(iii*50+400)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])









