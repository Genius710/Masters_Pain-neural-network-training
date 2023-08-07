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



mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong01Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLong0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong11Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLong1']),0,1)

InputLong1 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
LabelLong1 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

DataMatrix =[]     
        

[x] = LabelLong1.shape
LabelLong1C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLong1[iii] ==0:
        LabelLong1C[iii,0] = 1
        LabelLong1C[iii,1] = 0


    elif LabelLong1[iii] ==1: 
        LabelLong1C[iii,0] = 0
        LabelLong1C[iii,1] = 1
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV01Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLongV0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV11Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLongV1']),0,1)

InputLongV1 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
LabelLongV1 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

DataMatrix =[]     
        

[x] = LabelLongV1.shape
LabelLongV1C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongV1[iii] ==0:
        LabelLongV1C[iii,0] = 1
        LabelLongV1C[iii,1] = 0


    elif LabelLongV1[iii] ==1: 
        LabelLongV1C[iii,0] = 0
        LabelLongV1C[iii,1] = 1
        
InputLong1CNN = InputLong1.reshape(-1,VECL*multi,1)
LabelLong1CCNN = LabelLong1C.reshape(-1,2,1)


InputLongV1CNN = InputLongV1.reshape(-1,VECL*multi,1)
LabelLongV1CCNN = LabelLongV1C.reshape(-1,2,1)

InputLong1LSTM = InputLong1.reshape(-1,multi,VECL)
LabelLong1CLSTM = LabelLong1C.reshape(-1,2,1)


InputLongV1LSTM = InputLongV1.reshape(-1,multi,VECL)
LabelLongV1CLSTM = LabelLongV1C.reshape(-1,2,1)        
        
        
        
multi =5        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong05Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLong0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong15Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLong1']),0,1)

InputLong5 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
LabelLong5 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

DataMatrix =[]     
        

[x] = LabelLong5.shape
LabelLong5C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLong5[iii] ==0:
        LabelLong5C[iii,0] = 1
        LabelLong5C[iii,1] = 0


    elif LabelLong5[iii] ==1: 
        LabelLong5C[iii,0] = 0
        LabelLong5C[iii,1] = 1
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV05Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLongV0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV15Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLongV1']),0,1)

InputLongV5 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
LabelLongV5 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

DataMatrix =[]     
        

[x] = LabelLongV5.shape
LabelLongV5C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongV5[iii] ==0:
        LabelLongV5C[iii,0] = 1
        LabelLongV5C[iii,1] = 0


    elif LabelLongV5[iii] ==1: 
        LabelLongV5C[iii,0] = 0
        LabelLongV5C[iii,1] = 1
        
InputLong5CNN = InputLong5.reshape(-1,VECL*multi,1)
LabelLong5CCNN = LabelLong5C.reshape(-1,2,1)


InputLongV5CNN = InputLongV5.reshape(-1,VECL*multi,1)
LabelLongV5CCNN = LabelLongV5C.reshape(-1,2,1)

InputLong5LSTM = InputLong5.reshape(-1,multi,VECL)
LabelLong5CLSTM = LabelLong5C.reshape(-1,2,1)


InputLongV5LSTM = InputLongV5.reshape(-1,multi,VECL)
LabelLongV5CLSTM = LabelLongV5C.reshape(-1,2,1)                



multi =12        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong012Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLong0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong112Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLong1']),0,1)

InputLong12 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
LabelLong12 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

DataMatrix =[]     
        

[x] = LabelLong12.shape
LabelLong12C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLong12[iii] ==0:
        LabelLong12C[iii,0] = 1
        LabelLong12C[iii,1] = 0


    elif LabelLong12[iii] ==1: 
        LabelLong12C[iii,0] = 0
        LabelLong12C[iii,1] = 1
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV012Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLongV0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV112Features_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix1 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLongV1']),0,1)

InputLongV12 = np.vstack(( DataMatrix0[:,2:2+multi*VECL],DataMatrix1[:,2:2+multi*VECL]))
LabelLongV12 = np.hstack(( DataMatrix0[:,0],DataMatrix1[:,0]))

DataMatrix =[]     
        

[x] = LabelLongV12.shape
LabelLongV12C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongV12[iii] ==0:
        LabelLongV12C[iii,0] = 1
        LabelLongV12C[iii,1] = 0


    elif LabelLongV12[iii] ==1: 
        LabelLongV12C[iii,0] = 0
        LabelLongV12C[iii,1] = 1
        
InputLong12CNN = InputLong12.reshape(-1,VECL*multi,1)
LabelLong12CCNN = LabelLong12C.reshape(-1,2,1)


InputLongV12CNN = InputLongV12.reshape(-1,VECL*multi,1)
LabelLongV12CCNN = LabelLongV12C.reshape(-1,2,1)

InputLong12LSTM = InputLong12.reshape(-1,multi,VECL)
LabelLong12CLSTM = LabelLong12C.reshape(-1,2,1)


InputLongV12LSTM = InputLongV12.reshape(-1,multi,VECL)
LabelLongV12CLSTM = LabelLongV12C.reshape(-1,2,1)    

# multi =1
eep = 11
epoch_no = 50
size = 64;
model = tf.keras.models.load_model('RawSignalLong1Features_MLP_v1_recog/1')

model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputLong1 ,LabelLong1C ,epochs = epoch_no,batch_size =size,validation_data =(InputLongV1 ,LabelLongV1C ))
    model.save(pjoin('RawSignalLong1Features_MLP_v1_recog/',str(iii*50+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
   
    
model = tf.keras.models.load_model('RawSignalLong1Features_LSTM_v1_recog/1')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputLong1LSTM,LabelLong1CLSTM,epochs = epoch_no,batch_size =size,validation_data =(InputLongV1LSTM,LabelLongV1CLSTM))
    model.save(pjoin('RawSignalLong1Features_LSTM_v1_recog/',str(iii*50+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])    
    

model = tf.keras.models.load_model('RawSignalLong1Features_CNN_v1_recog/1')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputLong1CNN,LabelLong1CCNN,epochs = epoch_no,batch_size =size,validation_data =(InputLongV1CNN,LabelLongV1CCNN))
    model.save(pjoin('RawSignalLong1Features_CNN_v1_recog/',str(iii*50+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
    
    
multi = 5


    
    
    
model = tf.keras.models.load_model('RawSignalLong5Features_MLP_v1_recog/1')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputLong5 ,LabelLong5C ,epochs = epoch_no,batch_size =size,validation_data =(InputLongV5 ,LabelLongV5C ))
    model.save(pjoin('RawSignalLong5Features_MLP_v1_recog/',str(iii*50+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])


model = tf.keras.models.load_model('RawSignalLong5Features_CNN_v1_recog/1')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep-2):
    history = model.fit(InputLong5CNN,LabelLong5CCNN,epochs = epoch_no,batch_size =size,validation_data =(InputLongV5CNN,LabelLongV5CCNN))
    model.save(pjoin('RawSignalLong5Features_CNN_v1_recog/',str(iii*50+100)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])

model = tf.keras.models.load_model('RawSignalLong5Features_LSTM_v1_recog/1')
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(3,eep):
    history = model.fit(InputLong5LSTM,LabelLong5CLSTM,epochs = epoch_no,batch_size =size,validation_data =(InputLongV5LSTM,LabelLongV5CLSTM))
    model.save(pjoin('RawSignalLong5Features_LSTM_v1_recog/',str(iii*50+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])



multi = 12
# eep = 2
# epoch_no = 1
    

model = tf.keras.models.load_model('RawSignalLong12Features_CNN_v1_recog/1')

model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputLong12CNN,LabelLong12CCNN,epochs = epoch_no,batch_size =size,validation_data =(InputLongV12CNN,LabelLongV12CCNN))
    model.save(pjoin('RawSignalLong12Features_CNN_v1_recog/',str(iii*50+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
    

model = tf.keras.models.load_model('RawSignalLong12Features_MLP_v1_recog/1')

model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(5,eep):
    history = model.fit(InputLong12 ,LabelLong12C ,epochs = epoch_no,batch_size =size,validation_data =(InputLongV12 ,LabelLongV12C ))
    model.save(pjoin('RawSignalLong12Features_MLP_v1_recog/',str(iii*50+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])

model = tf.keras.models.load_model('RawSignalLong12Features_LSTM_v1_recog/1')

model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputLong12LSTM,LabelLong12CLSTM,epochs = epoch_no,batch_size =size,validation_data =(InputLongV12LSTM,LabelLongV12CLSTM))
    model.save(pjoin('RawSignalLong12Features_LSTM_v1_recog/',str(iii*50+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])





