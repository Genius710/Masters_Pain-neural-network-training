# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:17:11 2021

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


# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV1_triclass.mat')
# mat_contents = sio.loadmat(mat_fname)
# DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
# InputMeanTriclassV1 = DataMatrix[:,2:2+multi*VECL]
# LabelMeanTriclassV1 = DataMatrix[:,0]

# DataMatrix =[]     
        
        

# [x] = LabelMeanTriclassV1.shape
# LabelMeanTriclassV1C = np.empty([x,3], dtype=int)
       
# for iii in range(0,x):
    
#     if LabelMeanTriclassV1[iii] ==0:
#         LabelMeanTriclassV1C[iii,0] = 1
#         LabelMeanTriclassV1C[iii,1] = 0
#         LabelMeanTriclassV1C[iii,2] = 0


#     elif LabelMeanTriclassV1[iii] ==1: 
#         LabelMeanTriclassV1C[iii,0] = 0
#         LabelMeanTriclassV1C[iii,1] = 1
#         LabelMeanTriclassV1C[iii,2] = 0
        
#     elif LabelMeanTriclassV1[iii] ==2: 
#         LabelMeanTriclassV1C[iii,0] = 0
#         LabelMeanTriclassV1C[iii,1] = 0
#         LabelMeanTriclassV1C[iii,2] = 1
        

# InputMeanTriclassV1CNN = InputMeanTriclassV1.reshape(-1,VECL*multi,1)
# LabelMeanTriclassV1CCNN = LabelMeanTriclassV1C.reshape(-1,3,1)

# InputMeanTriclassV1LSTM = InputMeanTriclassV1.reshape(-1,multi,VECL)
# LabelMeanTriclassV1CLSTM = LabelMeanTriclassV1C.reshape(-1,3,1)        
        
        
        
multi =1        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV5_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
InputMeanTriclassV5 = DataMatrix[:,2:2+multi*VECL]
LabelMeanTriclassV5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMeanTriclassV5.shape
LabelMeanTriclassV5C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanTriclassV5[iii] ==0:
        LabelMeanTriclassV5C[iii,0] = 1
        LabelMeanTriclassV5C[iii,1] = 0
        LabelMeanTriclassV5C[iii,2] = 0


    elif LabelMeanTriclassV5[iii] ==1: 
        LabelMeanTriclassV5C[iii,0] = 0
        LabelMeanTriclassV5C[iii,1] = 1
        LabelMeanTriclassV5C[iii,2] = 0
    elif LabelMeanTriclassV5[iii] ==2: 
        LabelMeanTriclassV5C[iii,0] = 0
        LabelMeanTriclassV5C[iii,1] = 0
        LabelMeanTriclassV5C[iii,2] = 1
        


InputMeanTriclassV5CNN = InputMeanTriclassV5.reshape(-1,VECL*multi,1)
LabelMeanTriclassV5CCNN = LabelMeanTriclassV5C.reshape(-1,3,1)


InputMeanTriclassV5LSTM = InputMeanTriclassV5.reshape(-1,multi,VECL)
LabelMeanTriclassV5CLSTM = LabelMeanTriclassV5C.reshape(-1,3,1)                



multi =1      
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV12_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
InputMeanTriclassV12 = DataMatrix[:,2:2+multi*VECL]
LabelMeanTriclassV12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMeanTriclassV12.shape
LabelMeanTriclassV12C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanTriclassV12[iii] ==0:
        LabelMeanTriclassV12C[iii,0] = 1
        LabelMeanTriclassV12C[iii,1] = 0
        LabelMeanTriclassV12C[iii,2] = 0


    elif LabelMeanTriclassV12[iii] ==1: 
        LabelMeanTriclassV12C[iii,0] = 0
        LabelMeanTriclassV12C[iii,1] = 1
        LabelMeanTriclassV12C[iii,2] = 0
    elif LabelMeanTriclassV12[iii] ==2: 
        LabelMeanTriclassV12C[iii,0] = 0
        LabelMeanTriclassV12C[iii,1] = 0
        LabelMeanTriclassV12C[iii,2] = 1


InputMeanTriclassV12CNN = InputMeanTriclassV12.reshape(-1,VECL*multi,1)
LabelMeanTriclassV12CCNN = LabelMeanTriclassV12C.reshape(-1,3,1)


InputMeanTriclassV12LSTM = InputMeanTriclassV12.reshape(-1,multi,VECL)
LabelMeanTriclassV12CLSTM = LabelMeanTriclassV12C.reshape(-1,3,1)  






multi = 1
VECL = 500


# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV1_recog.mat')
# mat_contents = sio.loadmat(mat_fname)
# DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
# InputMeanRecogV1 = DataMatrix[:,2:2+multi*VECL]
# LabelMeanRecogV1 = DataMatrix[:,0]

# DataMatrix =[]     
        
        

# [x] = LabelMeanRecogV1.shape
# LabelMeanRecogV1C = np.empty([x,3], dtype=int)
       
# for iii in range(0,x):
    
#     if LabelMeanRecogV1[iii] ==0:
#         LabelMeanRecogV1C[iii,0] = 1
#         LabelMeanRecogV1C[iii,1] = 0
#         LabelMeanRecogV1C[iii,2] = 0


#     elif LabelMeanRecogV1[iii] ==1: 
#         LabelMeanRecogV1C[iii,0] = 0
#         LabelMeanRecogV1C[iii,1] = 1
#         LabelMeanRecogV1C[iii,2] = 0
        
#     elif LabelMeanRecogV1[iii] ==2: 
#         LabelMeanRecogV1C[iii,0] = 0
#         LabelMeanRecogV1C[iii,1] = 0
#         LabelMeanRecogV1C[iii,2] = 1
        

# InputMeanRecogV1CNN = InputMeanRecogV1.reshape(-1,VECL*multi,1)
# LabelMeanRecogV1CCNN = LabelMeanRecogV1C.reshape(-1,3,1)

# InputMeanRecogV1LSTM = InputMeanRecogV1.reshape(-1,multi,VECL)
# LabelMeanRecogV1CLSTM = LabelMeanRecogV1C.reshape(-1,3,1)        
        
        
        
multi =1       
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV5_recog.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
InputMeanRecogV5 = DataMatrix[:,2:2+multi*VECL]
LabelMeanRecogV5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMeanRecogV5.shape
LabelMeanRecogV5C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanRecogV5[iii] ==0:
        LabelMeanRecogV5C[iii,0] = 1
        LabelMeanRecogV5C[iii,1] = 0



    elif LabelMeanRecogV5[iii] ==1: 
        LabelMeanRecogV5C[iii,0] = 0
        LabelMeanRecogV5C[iii,1] = 1

        


InputMeanRecogV5CNN = InputMeanRecogV5.reshape(-1,VECL*multi,1)
LabelMeanRecogV5CCNN = LabelMeanRecogV5C.reshape(-1,2,1)


InputMeanRecogV5LSTM = InputMeanRecogV5.reshape(-1,multi,VECL)
LabelMeanRecogV5CLSTM = LabelMeanRecogV5C.reshape(-1,2,1)                



multi =1     
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV12_recog.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
InputMeanRecogV12 = DataMatrix[:,2:2+multi*VECL]
LabelMeanRecogV12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMeanRecogV12.shape
LabelMeanRecogV12C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanRecogV12[iii] ==0:
        LabelMeanRecogV12C[iii,0] = 1
        LabelMeanRecogV12C[iii,1] = 0



    elif LabelMeanRecogV12[iii] ==1: 
        LabelMeanRecogV12C[iii,0] = 0
        LabelMeanRecogV12C[iii,1] = 1



InputMeanRecogV12CNN = InputMeanRecogV12.reshape(-1,VECL*multi,1)
LabelMeanRecogV12CCNN = LabelMeanRecogV12C.reshape(-1,2,1)


InputMeanRecogV12LSTM = InputMeanRecogV12.reshape(-1,multi,VECL)
LabelMeanRecogV12CLSTM = LabelMeanRecogV12C.reshape(-1,2,1)  






path = 'Trained/RawSignalMean5_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Mean5_results.txt'),model.predict(InputMeanTriclassV5))

path = 'Trained/RawSignalMean5_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'CNN_Mean5_results.txt'),model.predict(InputMeanTriclassV5CNN))

path = 'Trained/RawSignalMean5_LSTM_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Mean5_results.txt'),model.predict(InputMeanTriclassV5LSTM))    



# path = 'Trained/RawSignalMean12_CNN_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])  
# np.savetxt(pjoin(path,'CNN_Mean12_results.txt'),model.predict(InputMeanTriclassV12CNN))

# path = 'Trained/RawSignalMean12_MLP_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'MLP_Mean12_results.txt'),model.predict(InputMeanTriclassV12))

# path = 'Trained/RawSignalMean12_LSTM_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Mean12_results.txt'),model.predict(InputMeanTriclassV12LSTM))



# path = 'Trained/RawSignalMean5_MLP_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'MLP_Mean5_results.txt'),model.predict(InputMeanRecogV5))

# path = 'Trained/RawSignalMean5_CNN_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'CNN_Mean5_results.txt'),model.predict(InputMeanRecogV5CNN))

# path = 'Trained/RawSignalMean5_LSTM_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Mean5_results.txt'),model.predict(InputMeanRecogV5LSTM))    



# path = 'Trained/RawSignalMean12_CNN_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])  
# np.savetxt(pjoin(path,'CNN_Mean12_results.txt'),model.predict(InputMeanRecogV12CNN))

# path = 'Trained/RawSignalMean12_MLP_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'MLP_Mean12_results.txt'),model.predict(InputMeanRecogV12))

# path = 'Trained/RawSignalMean12_LSTM_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Mean12_results.txt'),model.predict(InputMeanRecogV12LSTM))









