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



        
        
multi =1    
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMean5Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixMean'])


InputMean5 = DataMatrix[:,2:2+VECL*multi]
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
        
    elif LabelMean5[iii] ==2: 
        LabelMean5C[iii,0] = 0
        LabelMean5C[iii,1] = 0
        LabelMean5C[iii,2] = 1
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMeanV5Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixMeanV'])


InputMeanV5 = DataMatrix[:,2:2+VECL*multi]
LabelMeanV5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMeanV5.shape
LabelMeanV5C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanV5[iii] ==0:
        LabelMeanV5C[iii,0] = 1
        LabelMeanV5C[iii,1] = 0
        LabelMeanV5C[iii,2] = 0


    elif LabelMeanV5[iii] ==1: 
        LabelMeanV5C[iii,0] = 0
        LabelMeanV5C[iii,1] = 1
        LabelMeanV5C[iii,2] = 0
        
    elif LabelMeanV5[iii] ==2: 
        LabelMeanV5C[iii,0] = 0
        LabelMeanV5C[iii,1] = 0
        LabelMeanV5C[iii,2] = 1
        
InputMean5CNN = InputMean5.reshape(-1,VECL*multi,1)
LabelMean5CCNN = LabelMean5C.reshape(-1,3,1)


InputMeanV5CNN = InputMeanV5.reshape(-1,VECL*multi,1)
LabelMeanV5CCNN = LabelMeanV5C.reshape(-1,3,1)

InputMean5LSTM = InputMean5.reshape(-1,multi,VECL)
LabelMean5CLSTM = LabelMean5C.reshape(-1,3,1)


InputMeanV5LSTM = InputMeanV5.reshape(-1,multi,VECL)
LabelMeanV5CLSTM = LabelMeanV5C.reshape(-1,3,1)                



multi =1   
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMean12Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixMean'])


InputMean12 = DataMatrix[:,2:2+VECL*multi]
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
        
    elif LabelMean12[iii] ==2: 
        LabelMean12C[iii,0] = 0
        LabelMean12C[iii,1] = 0
        LabelMean12C[iii,2] = 1
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMeanV12Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixMeanV'])


InputMeanV12 = DataMatrix[:,2:2+VECL*multi]
LabelMeanV12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMeanV12.shape
LabelMeanV12C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanV12[iii] ==0:
        LabelMeanV12C[iii,0] = 1
        LabelMeanV12C[iii,1] = 0
        LabelMeanV12C[iii,2] = 0


    elif LabelMeanV12[iii] ==1: 
        LabelMeanV12C[iii,0] = 0
        LabelMeanV12C[iii,1] = 1
        LabelMeanV12C[iii,2] = 0
        
    elif LabelMeanV12[iii] ==2: 
        LabelMeanV12C[iii,0] = 0
        LabelMeanV12C[iii,1] = 0
        LabelMeanV12C[iii,2] = 1
        
InputMean12CNN = InputMean12.reshape(-1,VECL*multi,1)
LabelMean12CCNN = LabelMean12C.reshape(-1,3,1)


InputMeanV12CNN = InputMeanV12.reshape(-1,VECL*multi,1)
LabelMeanV12CCNN = LabelMeanV12C.reshape(-1,3,1)

InputMean12LSTM = InputMean12.reshape(-1,multi,VECL)
LabelMean12CLSTM = LabelMean12C.reshape(-1,3,1)


InputMeanV12LSTM = InputMeanV12.reshape(-1,multi,VECL)
LabelMeanV12CLSTM = LabelMeanV12C.reshape(-1,3,1)    

epoch_no =1


    
model = tf.keras.models.load_model('Trained/RawSignalMean5Features_MLP_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputMean5 ,LabelMean5C ,epochs = epoch_no,batch_size =128,validation_data =(InputMeanV5 ,LabelMeanV5C ))


model = tf.keras.models.load_model('Trained/RawSignalMean5Features_CNN_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputMean5CNN,LabelMean5CCNN,epochs = epoch_no,batch_size =64,validation_data =(InputMeanV5CNN,LabelMeanV5CCNN))


model = tf.keras.models.load_model('Trained/RawSignalMean5Features_LSTM_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputMean5LSTM,LabelMean5CLSTM,epochs = epoch_no,batch_size =32,validation_data =(InputMeanV5LSTM,LabelMeanV5CLSTM))
   

model = tf.keras.models.load_model('Trained/RawSignalMean12Features_CNN_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputMean12CNN,LabelMean12CCNN,epochs = epoch_no,batch_size =32,validation_data =(InputMeanV12CNN,LabelMeanV12CCNN))
   

model = tf.keras.models.load_model('Trained/RawSignalMean12Features_MLP_v1_triclass/500')
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputMean12 ,LabelMean12C ,epochs = epoch_no,batch_size =256,validation_data =(InputMeanV12 ,LabelMeanV12C ))

model = tf.keras.models.load_model('Trained/RawSignalMean12Features_LSTM_v1_triclass/500')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(InputMean12LSTM,LabelMean12CLSTM,epochs = epoch_no,batch_size =16,validation_data =(InputMeanV12LSTM,LabelMeanV12CLSTM))



