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
VECL = 68


mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLongV1Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLongV'])
InputLongFeaturesTriclassV1 = DataMatrix[:,2:2+multi*VECL]
LabelLongFeaturesTriclassV1 = DataMatrix[:,0]

DataMatrix =[]     
        
        

[x] = LabelLongFeaturesTriclassV1.shape
LabelLongFeaturesTriclassV1C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongFeaturesTriclassV1[iii] ==0:
        LabelLongFeaturesTriclassV1C[iii,0] = 1
        LabelLongFeaturesTriclassV1C[iii,1] = 0
        LabelLongFeaturesTriclassV1C[iii,2] = 0


    elif LabelLongFeaturesTriclassV1[iii] ==1: 
        LabelLongFeaturesTriclassV1C[iii,0] = 0
        LabelLongFeaturesTriclassV1C[iii,1] = 1
        LabelLongFeaturesTriclassV1C[iii,2] = 0
        
    elif LabelLongFeaturesTriclassV1[iii] ==2: 
        LabelLongFeaturesTriclassV1C[iii,0] = 0
        LabelLongFeaturesTriclassV1C[iii,1] = 0
        LabelLongFeaturesTriclassV1C[iii,2] = 1
        

InputLongFeaturesTriclassV1CNN = InputLongFeaturesTriclassV1.reshape(-1,VECL*multi,1)
LabelLongFeaturesTriclassV1CCNN = LabelLongFeaturesTriclassV1C.reshape(-1,3,1)

InputLongFeaturesTriclassV1LSTM = InputLongFeaturesTriclassV1.reshape(-1,multi,VECL)
LabelLongFeaturesTriclassV1CLSTM = LabelLongFeaturesTriclassV1C.reshape(-1,3,1)        
        
        
        
multi =5        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLongV5Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLongV'])
InputLongFeaturesTriclassV5 = DataMatrix[:,2:2+multi*VECL]
LabelLongFeaturesTriclassV5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLongFeaturesTriclassV5.shape
LabelLongFeaturesTriclassV5C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongFeaturesTriclassV5[iii] ==0:
        LabelLongFeaturesTriclassV5C[iii,0] = 1
        LabelLongFeaturesTriclassV5C[iii,1] = 0
        LabelLongFeaturesTriclassV5C[iii,2] = 0


    elif LabelLongFeaturesTriclassV5[iii] ==1: 
        LabelLongFeaturesTriclassV5C[iii,0] = 0
        LabelLongFeaturesTriclassV5C[iii,1] = 1
        LabelLongFeaturesTriclassV5C[iii,2] = 0
    elif LabelLongFeaturesTriclassV5[iii] ==2: 
        LabelLongFeaturesTriclassV5C[iii,0] = 0
        LabelLongFeaturesTriclassV5C[iii,1] = 0
        LabelLongFeaturesTriclassV5C[iii,2] = 1
        


InputLongFeaturesTriclassV5CNN = InputLongFeaturesTriclassV5.reshape(-1,VECL*multi,1)
LabelLongFeaturesTriclassV5CCNN = LabelLongFeaturesTriclassV5C.reshape(-1,3,1)


InputLongFeaturesTriclassV5LSTM = InputLongFeaturesTriclassV5.reshape(-1,multi,VECL)
LabelLongFeaturesTriclassV5CLSTM = LabelLongFeaturesTriclassV5C.reshape(-1,3,1)                



multi =12        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLongV12Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLongV'])
InputLongFeaturesTriclassV12 = DataMatrix[:,2:2+multi*VECL]
LabelLongFeaturesTriclassV12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLongFeaturesTriclassV12.shape
LabelLongFeaturesTriclassV12C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongFeaturesTriclassV12[iii] ==0:
        LabelLongFeaturesTriclassV12C[iii,0] = 1
        LabelLongFeaturesTriclassV12C[iii,1] = 0
        LabelLongFeaturesTriclassV12C[iii,2] = 0


    elif LabelLongFeaturesTriclassV12[iii] ==1: 
        LabelLongFeaturesTriclassV12C[iii,0] = 0
        LabelLongFeaturesTriclassV12C[iii,1] = 1
        LabelLongFeaturesTriclassV12C[iii,2] = 0
    elif LabelLongFeaturesTriclassV12[iii] ==2: 
        LabelLongFeaturesTriclassV12C[iii,0] = 0
        LabelLongFeaturesTriclassV12C[iii,1] = 0
        LabelLongFeaturesTriclassV12C[iii,2] = 1


InputLongFeaturesTriclassV12CNN = InputLongFeaturesTriclassV12.reshape(-1,VECL*multi,1)
LabelLongFeaturesTriclassV12CCNN = LabelLongFeaturesTriclassV12C.reshape(-1,3,1)


InputLongFeaturesTriclassV12LSTM = InputLongFeaturesTriclassV12.reshape(-1,multi,VECL)
LabelLongFeaturesTriclassV12CLSTM = LabelLongFeaturesTriclassV12C.reshape(-1,3,1)  






multi = 1
VECL = 68


mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLongV1Features_recog.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLongV'])
InputLongFeaturesRecogV1 = DataMatrix[:,2:2+multi*VECL]
LabelLongFeaturesRecogV1 = DataMatrix[:,0]

DataMatrix =[]     
        
        

[x] = LabelLongFeaturesRecogV1.shape
LabelLongFeaturesRecogV1C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongFeaturesRecogV1[iii] ==0:
        LabelLongFeaturesRecogV1C[iii,0] = 1
        LabelLongFeaturesRecogV1C[iii,1] = 0



    elif LabelLongFeaturesRecogV1[iii] ==1: 
        LabelLongFeaturesRecogV1C[iii,0] = 0
        LabelLongFeaturesRecogV1C[iii,1] = 1

        

InputLongFeaturesRecogV1CNN = InputLongFeaturesRecogV1.reshape(-1,VECL*multi,1)
LabelLongFeaturesRecogV1CCNN = LabelLongFeaturesRecogV1C.reshape(-1,2,1)

InputLongFeaturesRecogV1LSTM = InputLongFeaturesRecogV1.reshape(-1,multi,VECL)
LabelLongFeaturesRecogV1CLSTM = LabelLongFeaturesRecogV1C.reshape(-1,2,1)        
        
        
        
multi =5        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLongV5Features_recog.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLongV'])
InputLongFeaturesRecogV5 = DataMatrix[:,2:2+multi*VECL]
LabelLongFeaturesRecogV5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLongFeaturesRecogV5.shape
LabelLongFeaturesRecogV5C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongFeaturesRecogV5[iii] ==0:
        LabelLongFeaturesRecogV5C[iii,0] = 1
        LabelLongFeaturesRecogV5C[iii,1] = 0



    elif LabelLongFeaturesRecogV5[iii] ==1: 
        LabelLongFeaturesRecogV5C[iii,0] = 0
        LabelLongFeaturesRecogV5C[iii,1] = 1




InputLongFeaturesRecogV5CNN = InputLongFeaturesRecogV5.reshape(-1,VECL*multi,1)
LabelLongFeaturesRecogV5CCNN = LabelLongFeaturesRecogV5C.reshape(-1,2,1)


InputLongFeaturesRecogV5LSTM = InputLongFeaturesRecogV5.reshape(-1,multi,VECL)
LabelLongFeaturesRecogV5CLSTM = LabelLongFeaturesRecogV5C.reshape(-1,2,1)                



multi =12        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLongV12Features_recog.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLongV'])
InputLongFeaturesRecogV12 = DataMatrix[:,2:2+multi*VECL]
LabelLongFeaturesRecogV12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLongFeaturesRecogV12.shape
LabelLongFeaturesRecogV12C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongFeaturesRecogV12[iii] ==0:
        LabelLongFeaturesRecogV12C[iii,0] = 1
        LabelLongFeaturesRecogV12C[iii,1] = 0



    elif LabelLongFeaturesRecogV12[iii] ==1: 
        LabelLongFeaturesRecogV12C[iii,0] = 0
        LabelLongFeaturesRecogV12C[iii,1] = 1




InputLongFeaturesRecogV12CNN = InputLongFeaturesRecogV12.reshape(-1,VECL*multi,1)
LabelLongFeaturesRecogV12CCNN = LabelLongFeaturesRecogV12C.reshape(-1,2,1)


InputLongFeaturesRecogV12LSTM = InputLongFeaturesRecogV12.reshape(-1,multi,VECL)
LabelLongFeaturesRecogV12CLSTM = LabelLongFeaturesRecogV12C.reshape(-1,2,1)  




# path = 'Trained/RawSignalLong1Features_MLP_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'MLP_Long1_results.txt'),model.predict(InputLongFeaturesTriclassV1))

# path = 'Trained/RawSignalLong1Features_CNN_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'CNN_Long1_results.txt'),model.predict(InputLongFeaturesTriclassV1CNN))

# path = 'Trained/RawSignalLong1Features_LSTM_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Long1_results.txt'),model.predict(InputLongFeaturesTriclassV1LSTM))    


# path = 'Trained/RawSignalLong5Features_MLP_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'MLP_Long5_results.txt'),model.predict(InputLongFeaturesTriclassV5))

# path = 'Trained/RawSignalLong5Features_CNN_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'CNN_Long5_results.txt'),model.predict(InputLongFeaturesTriclassV5CNN))

# path = 'Trained/RawSignalLong5Features_LSTM_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Long5_results.txt'),model.predict(InputLongFeaturesTriclassV5LSTM))    



# path = 'Trained/RawSignalLong12Features_CNN_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])  
# np.savetxt(pjoin(path,'CNN_Long12_results.txt'),model.predict(InputLongFeaturesTriclassV12CNN))

# path = 'Trained/RawSignalLong12Features_MLP_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'MLP_Long12_results.txt'),model.predict(InputLongFeaturesTriclassV12))

# path = 'Trained/RawSignalLong12Features_LSTM_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Long12_results.txt'),model.predict(InputLongFeaturesTriclassV12LSTM))





# path = 'Trained/RawSignalLong1Features_MLP_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'MLP_Long1_results.txt'),model.predict(InputLongFeaturesRecogV1))

# path = 'Trained/RawSignalLong1Features_CNN_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'CNN_Long1_results.txt'),model.predict(InputLongFeaturesRecogV1CNN))

# path = 'Trained/RawSignalLong1Features_LSTM_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Long1_results.txt'),model.predict(InputLongFeaturesRecogV1LSTM))   



# path = 'Trained/RawSignalLong5Features_MLP_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'MLP_Long5_results.txt'),model.predict(InputLongFeaturesRecogV5))

# path = 'Trained/RawSignalLong5Features_CNN_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'CNN_Long5_results.txt'),model.predict(InputLongFeaturesRecogV5CNN))

# path = 'Trained/RawSignalLong5Features_LSTM_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Long5_results.txt'),model.predict(InputLongFeaturesRecogV5LSTM))    



path = 'Trained/RawSignalLong12Features_CNN_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'210')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])  
np.savetxt(pjoin(path,'CNN_Long12_results.txt'),model.predict(InputLongFeaturesRecogV12CNN))

path = 'Trained/RawSignalLong12Features_MLP_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Long12_results.txt'),model.predict(InputLongFeaturesRecogV12))

path = 'Trained/RawSignalLong12Features_LSTM_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Long12_results.txt'),model.predict(InputLongFeaturesRecogV12LSTM))











