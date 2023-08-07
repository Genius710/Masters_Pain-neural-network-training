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
VECL = 500



mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong1_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLong'])


InputLong1 = DataMatrix[:,2:2+VECL]
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
        
    elif LabelLong1[iii] ==2: 
        LabelLong1C[iii,0] = 0
        LabelLong1C[iii,1] = 0
        LabelLong1C[iii,2] = 1
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV1_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLongV'])


InputLongV1 = DataMatrix[:,2:2+VECL]
LabelLongV1 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLongV1.shape
LabelLongV1C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongV1[iii] ==0:
        LabelLongV1C[iii,0] = 1
        LabelLongV1C[iii,1] = 0
        LabelLongV1C[iii,2] = 0


    elif LabelLongV1[iii] ==1: 
        LabelLongV1C[iii,0] = 0
        LabelLongV1C[iii,1] = 1
        LabelLongV1C[iii,2] = 0
        
    elif LabelLongV1[iii] ==2: 
        LabelLongV1C[iii,0] = 0
        LabelLongV1C[iii,1] = 0
        LabelLongV1C[iii,2] = 1
        
InputLong1CNN = InputLong1.reshape(-1,VECL*multi,1)
LabelLong1CCNN = LabelLong1C.reshape(-1,3,1)


InputLongV1CNN = InputLongV1.reshape(-1,VECL*multi,1)
LabelLongV1CCNN = LabelLongV1C.reshape(-1,3,1)

InputLong1LSTM = InputLong1.reshape(-1,multi,VECL)
LabelLong1CLSTM = LabelLong1C.reshape(-1,3,1)


InputLongV1LSTM = InputLongV1.reshape(-1,multi,VECL)
LabelLongV1CLSTM = LabelLongV1C.reshape(-1,3,1)        
        
        
        
multi =5        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong5_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLong'])


InputLong5 = DataMatrix[:,2:2+VECL*multi]
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
        
    elif LabelLong5[iii] ==2: 
        LabelLong5C[iii,0] = 0
        LabelLong5C[iii,1] = 0
        LabelLong5C[iii,2] = 1
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV5_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLongV'])


InputLongV5 = DataMatrix[:,2:2+VECL*multi]
LabelLongV5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLongV5.shape
LabelLongV5C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongV5[iii] ==0:
        LabelLongV5C[iii,0] = 1
        LabelLongV5C[iii,1] = 0
        LabelLongV5C[iii,2] = 0


    elif LabelLongV5[iii] ==1: 
        LabelLongV5C[iii,0] = 0
        LabelLongV5C[iii,1] = 1
        LabelLongV5C[iii,2] = 0
        
    elif LabelLongV5[iii] ==2: 
        LabelLongV5C[iii,0] = 0
        LabelLongV5C[iii,1] = 0
        LabelLongV5C[iii,2] = 1
        
InputLong5CNN = InputLong5.reshape(-1,VECL*multi,1)
LabelLong5CCNN = LabelLong5C.reshape(-1,3,1)


InputLongV5CNN = InputLongV5.reshape(-1,VECL*multi,1)
LabelLongV5CCNN = LabelLongV5C.reshape(-1,3,1)

InputLong5LSTM = InputLong5.reshape(-1,multi,VECL)
LabelLong5CLSTM = LabelLong5C.reshape(-1,3,1)


InputLongV5LSTM = InputLongV5.reshape(-1,multi,VECL)
LabelLongV5CLSTM = LabelLongV5C.reshape(-1,3,1)                



multi =12        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong12_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLong'])


InputLong12 = DataMatrix[:,2:2+VECL*multi]
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
        
    elif LabelLong12[iii] ==2: 
        LabelLong12C[iii,0] = 0
        LabelLong12C[iii,1] = 0
        LabelLong12C[iii,2] = 1
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV12_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixLongV'])


InputLongV12 = DataMatrix[:,2:2+VECL*multi]
LabelLongV12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLongV12.shape
LabelLongV12C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongV12[iii] ==0:
        LabelLongV12C[iii,0] = 1
        LabelLongV12C[iii,1] = 0
        LabelLongV12C[iii,2] = 0


    elif LabelLongV12[iii] ==1: 
        LabelLongV12C[iii,0] = 0
        LabelLongV12C[iii,1] = 1
        LabelLongV12C[iii,2] = 0
        
    elif LabelLongV12[iii] ==2: 
        LabelLongV12C[iii,0] = 0
        LabelLongV12C[iii,1] = 0
        LabelLongV12C[iii,2] = 1
        
InputLong12CNN = InputLong12.reshape(-1,VECL*multi,1)
LabelLong12CCNN = LabelLong12C.reshape(-1,3,1)


InputLongV12CNN = InputLongV12.reshape(-1,VECL*multi,1)
LabelLongV12CCNN = LabelLongV12C.reshape(-1,3,1)

InputLong12LSTM = InputLong12.reshape(-1,multi,VECL)
LabelLong12CLSTM = LabelLong12C.reshape(-1,3,1)


InputLongV12LSTM = InputLongV12.reshape(-1,multi,VECL)
LabelLongV12CLSTM = LabelLongV12C.reshape(-1,3,1)    

epoch_no =1

model = tf.keras.models.load_model('Trained/RawSignalLong1_MLP_v1_triclass/500')

model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputLong1 ,LabelLong1C ,epochs = epoch_no,batch_size =128,validation_data =(InputLongV1 ,LabelLongV1C ))
   
    
model = tf.keras.models.load_model('Trained/RawSignalLong1_LSTM_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputLong1LSTM,LabelLong1CLSTM,epochs = epoch_no,batch_size =128,validation_data =(InputLongV1LSTM,LabelLongV1CLSTM))
 

model = tf.keras.models.load_model('Trained/RawSignalLong1_CNN_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputLong1CNN,LabelLong1CCNN,epochs = epoch_no,batch_size =128,validation_data =(InputLongV1CNN,LabelLongV1CCNN))

    
model = tf.keras.models.load_model('Trained/RawSignalLong5_MLP_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputLong5 ,LabelLong5C ,epochs = epoch_no,batch_size =128,validation_data =(InputLongV5 ,LabelLongV5C ))


model = tf.keras.models.load_model('Trained/RawSignalLong5_CNN_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputLong5CNN,LabelLong5CCNN,epochs = epoch_no,batch_size =64,validation_data =(InputLongV5CNN,LabelLongV5CCNN))


model = tf.keras.models.load_model('Trained/RawSignalLong5_LSTM_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputLong5LSTM,LabelLong5CLSTM,epochs = epoch_no,batch_size =32,validation_data =(InputLongV5LSTM,LabelLongV5CLSTM))
   

model = tf.keras.models.load_model('Trained/RawSignalLong12_CNN_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputLong12CNN,LabelLong12CCNN,epochs = epoch_no,batch_size =32,validation_data =(InputLongV12CNN,LabelLongV12CCNN))
   

model = tf.keras.models.load_model('Trained/RawSignalLong12_MLP_v1_triclass/500')
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputLong12 ,LabelLong12C ,epochs = epoch_no,batch_size =256,validation_data =(InputLongV12 ,LabelLongV12C ))

model = tf.keras.models.load_model('Trained/RawSignalLong12_LSTM_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputLong12LSTM,LabelLong12CLSTM,epochs = epoch_no,batch_size =16,validation_data =(InputLongV12LSTM,LabelLongV12CLSTM))



