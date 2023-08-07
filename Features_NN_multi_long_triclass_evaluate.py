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


mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLongV1Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLongV'])
InputLongV1 = DataMatrix[:,2:2+multi*VECL]
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


    elif LabelLongV1[iii] ==2 : 
        LabelLongV1C[iii,0] = 0
        LabelLongV1C[iii,1] = 0
        LabelLongV1C[iii,2] = 1
        

InputLongV1CNN = InputLongV1.reshape(-1,VECL*multi,1)
LabelLongV1CCNN = LabelLongV1C.reshape(-1,3,1)

InputLongV1LSTM = InputLongV1.reshape(-1,multi,VECL)
LabelLongV1CLSTM = LabelLongV1C.reshape(-1,3,1)        
        
        
        
multi =5        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLongV5Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLongV'])
InputLongV5 = DataMatrix[:,2:2+multi*VECL]
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


    elif LabelLongV5[iii] ==2 : 
        LabelLongV5C[iii,0] = 0
        LabelLongV5C[iii,1] = 0
        LabelLongV5C[iii,2] = 1
        


InputLongV5CNN = InputLongV5.reshape(-1,VECL*multi,1)
LabelLongV5CCNN = LabelLongV5C.reshape(-1,3,1)


InputLongV5LSTM = InputLongV5.reshape(-1,multi,VECL)
LabelLongV5CLSTM = LabelLongV5C.reshape(-1,3,1)                



multi =12        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLongV12Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLongV'])
InputLongV12 = DataMatrix[:,2:2+multi*VECL]
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


    elif LabelLongV12[iii] ==2 : 
        LabelLongV12C[iii,0] = 0
        LabelLongV12C[iii,1] = 0
        LabelLongV12C[iii,2] = 1


InputLongV12CNN = InputLongV12.reshape(-1,VECL*multi,1)
LabelLongV12CCNN = LabelLongV12C.reshape(-1,3,1)


InputLongV12LSTM = InputLongV12.reshape(-1,multi,VECL)
LabelLongV12CLSTM = LabelLongV12C.reshape(-1,3,1)    


path = 'RawSignalMean1Features_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'450'))
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Long1_results.txt'),model.predict(InputLongV1))
   
path = 'RawSignalMean1Features_LSTM_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'450')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Long1_results.txt'),model.predict(InputLongV1LSTM))   
 
path = 'RawSignalMean1Features_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'450')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'CNN_Long1_results.txt'),model.predict(InputLongV1CNN))
    

    
path = 'RawSignalLong5Features_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'450')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Long5_results.txt'),model.predict(InputLongV5))

path = 'RawSignalLong5Features_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'450')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'CNN_Long5_results.txt'),model.predict(InputLongV5CNN))

path = 'RawSignalLong5Features_LSTM_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'450')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Long5_results.txt'),model.predict(InputLongV5LSTM))    



path = 'RawSignalLong12Features_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'450')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])  
np.savetxt(pjoin(path,'CNN_Long12_results.txt'),model.predict(InputLongV12CNN))

path = 'RawSignalLong12Features_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'450')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Long12_results.txt'),model.predict(InputLongV12))

path = 'RawSignalLong12Features_LSTM_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'450')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Long12_results.txt'),model.predict(InputLongV12LSTM))

