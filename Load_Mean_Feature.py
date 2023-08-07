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


# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV1Features_triclass.mat')
# mat_contents = sio.loadmat(mat_fname)
# DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
# InputMeanFeaturesTriclassV1 = DataMatrix[:,2:2+multi*VECL]
# LabelMeanFeaturesTriclassV1 = DataMatrix[:,0]

# DataMatrix =[]     
        
        

# [x] = LabelMeanFeaturesTriclassV1.shape
# LabelMeanFeaturesTriclassV1C = np.empty([x,3], dtype=int)
       
# for iii in range(0,x):
    
#     if LabelMeanFeaturesTriclassV1[iii] ==0:
#         LabelMeanFeaturesTriclassV1C[iii,0] = 1
#         LabelMeanFeaturesTriclassV1C[iii,1] = 0
#         LabelMeanFeaturesTriclassV1C[iii,2] = 0


#     elif LabelMeanFeaturesTriclassV1[iii] ==1: 
#         LabelMeanFeaturesTriclassV1C[iii,0] = 0
#         LabelMeanFeaturesTriclassV1C[iii,1] = 1
#         LabelMeanFeaturesTriclassV1C[iii,2] = 0
        
#     elif LabelMeanFeaturesTriclassV1[iii] ==2: 
#         LabelMeanFeaturesTriclassV1C[iii,0] = 0
#         LabelMeanFeaturesTriclassV1C[iii,1] = 0
#         LabelMeanFeaturesTriclassV1C[iii,2] = 1
        

# InputMeanFeaturesTriclassV1CNN = InputMeanFeaturesTriclassV1.reshape(-1,VECL*multi,1)
# LabelMeanFeaturesTriclassV1CCNN = LabelMeanFeaturesTriclassV1C.reshape(-1,3,1)

# InputMeanFeaturesTriclassV1LSTM = InputMeanFeaturesTriclassV1.reshape(-1,multi,VECL)
# LabelMeanFeaturesTriclassV1CLSTM = LabelMeanFeaturesTriclassV1C.reshape(-1,3,1)        
        
        
        
multi =1     
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV5Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
InputMeanFeaturesTriclassV5 = DataMatrix[:,2:2+multi*VECL]
LabelMeanFeaturesTriclassV5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMeanFeaturesTriclassV5.shape
LabelMeanFeaturesTriclassV5C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanFeaturesTriclassV5[iii] ==0:
        LabelMeanFeaturesTriclassV5C[iii,0] = 1
        LabelMeanFeaturesTriclassV5C[iii,1] = 0
        LabelMeanFeaturesTriclassV5C[iii,2] = 0


    elif LabelMeanFeaturesTriclassV5[iii] ==1: 
        LabelMeanFeaturesTriclassV5C[iii,0] = 0
        LabelMeanFeaturesTriclassV5C[iii,1] = 1
        LabelMeanFeaturesTriclassV5C[iii,2] = 0
    elif LabelMeanFeaturesTriclassV5[iii] ==2: 
        LabelMeanFeaturesTriclassV5C[iii,0] = 0
        LabelMeanFeaturesTriclassV5C[iii,1] = 0
        LabelMeanFeaturesTriclassV5C[iii,2] = 1
        


InputMeanFeaturesTriclassV5CNN = InputMeanFeaturesTriclassV5.reshape(-1,VECL*multi,1)
LabelMeanFeaturesTriclassV5CCNN = LabelMeanFeaturesTriclassV5C.reshape(-1,3,1)


InputMeanFeaturesTriclassV5LSTM = InputMeanFeaturesTriclassV5.reshape(-1,multi,VECL)
LabelMeanFeaturesTriclassV5CLSTM = LabelMeanFeaturesTriclassV5C.reshape(-1,3,1)                



multi =1
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV12Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
InputMeanFeaturesTriclassV12 = DataMatrix[:,2:2+multi*VECL]
LabelMeanFeaturesTriclassV12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMeanFeaturesTriclassV12.shape
LabelMeanFeaturesTriclassV12C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanFeaturesTriclassV12[iii] ==0:
        LabelMeanFeaturesTriclassV12C[iii,0] = 1
        LabelMeanFeaturesTriclassV12C[iii,1] = 0
        LabelMeanFeaturesTriclassV12C[iii,2] = 0


    elif LabelMeanFeaturesTriclassV12[iii] ==1: 
        LabelMeanFeaturesTriclassV12C[iii,0] = 0
        LabelMeanFeaturesTriclassV12C[iii,1] = 1
        LabelMeanFeaturesTriclassV12C[iii,2] = 0
    elif LabelMeanFeaturesTriclassV12[iii] ==2: 
        LabelMeanFeaturesTriclassV12C[iii,0] = 0
        LabelMeanFeaturesTriclassV12C[iii,1] = 0
        LabelMeanFeaturesTriclassV12C[iii,2] = 1


InputMeanFeaturesTriclassV12CNN = InputMeanFeaturesTriclassV12.reshape(-1,VECL*multi,1)
LabelMeanFeaturesTriclassV12CCNN = LabelMeanFeaturesTriclassV12C.reshape(-1,3,1)


InputMeanFeaturesTriclassV12LSTM = InputMeanFeaturesTriclassV12.reshape(-1,multi,VECL)
LabelMeanFeaturesTriclassV12CLSTM = LabelMeanFeaturesTriclassV12C.reshape(-1,3,1)  






multi = 1
VECL = 68


# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV1Features_recog.mat')
# mat_contents = sio.loadmat(mat_fname)
# DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
# InputMeanFeaturesRecogV1 = DataMatrix[:,2:2+multi*VECL]
# LabelMeanFeaturesRecogV1 = DataMatrix[:,0]

# DataMatrix =[]     
        
        

# [x] = LabelMeanFeaturesRecogV1.shape
# LabelMeanFeaturesRecogV1C = np.empty([x,2], dtype=int)
       
# for iii in range(0,x):
    
#     if LabelMeanFeaturesRecogV1[iii] ==0:
#         LabelMeanFeaturesRecogV1C[iii,0] = 1
#         LabelMeanFeaturesRecogV1C[iii,1] = 0



#     elif LabelMeanFeaturesRecogV1[iii] ==1: 
#         LabelMeanFeaturesRecogV1C[iii,0] = 0
#         LabelMeanFeaturesRecogV1C[iii,1] = 1

        

# InputMeanFeaturesRecogV1CNN = InputMeanFeaturesRecogV1.reshape(-1,VECL*multi,1)
# LabelMeanFeaturesRecogV1CCNN = LabelMeanFeaturesRecogV1C.reshape(-1,2,1)

# InputMeanFeaturesRecogV1LSTM = InputMeanFeaturesRecogV1.reshape(-1,multi,VECL)
# LabelMeanFeaturesRecogV1CLSTM = LabelMeanFeaturesRecogV1C.reshape(-1,2,1)        
        
        
        
multi =1       
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV5Features_recog.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
InputMeanFeaturesRecogV5 = DataMatrix[:,2:2+multi*VECL]
LabelMeanFeaturesRecogV5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMeanFeaturesRecogV5.shape
LabelMeanFeaturesRecogV5C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanFeaturesRecogV5[iii] ==0:
        LabelMeanFeaturesRecogV5C[iii,0] = 1
        LabelMeanFeaturesRecogV5C[iii,1] = 0



    elif LabelMeanFeaturesRecogV5[iii] ==1: 
        LabelMeanFeaturesRecogV5C[iii,0] = 0
        LabelMeanFeaturesRecogV5C[iii,1] = 1




InputMeanFeaturesRecogV5CNN = InputMeanFeaturesRecogV5.reshape(-1,VECL*multi,1)
LabelMeanFeaturesRecogV5CCNN = LabelMeanFeaturesRecogV5C.reshape(-1,2,1)


InputMeanFeaturesRecogV5LSTM = InputMeanFeaturesRecogV5.reshape(-1,multi,VECL)
LabelMeanFeaturesRecogV5CLSTM = LabelMeanFeaturesRecogV5C.reshape(-1,2,1)                



multi =1     
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV12Features_recog.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixMeanV'])
InputMeanFeaturesRecogV12 = DataMatrix[:,2:2+multi*VECL]
LabelMeanFeaturesRecogV12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelMeanFeaturesRecogV12.shape
LabelMeanFeaturesRecogV12C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelMeanFeaturesRecogV12[iii] ==0:
        LabelMeanFeaturesRecogV12C[iii,0] = 1
        LabelMeanFeaturesRecogV12C[iii,1] = 0



    elif LabelMeanFeaturesRecogV12[iii] ==1: 
        LabelMeanFeaturesRecogV12C[iii,0] = 0
        LabelMeanFeaturesRecogV12C[iii,1] = 1




InputMeanFeaturesRecogV12CNN = InputMeanFeaturesRecogV12.reshape(-1,VECL*multi,1)
LabelMeanFeaturesRecogV12CCNN = LabelMeanFeaturesRecogV12C.reshape(-1,2,1)


InputMeanFeaturesRecogV12LSTM = InputMeanFeaturesRecogV12.reshape(-1,multi,VECL)
LabelMeanFeaturesRecogV12CLSTM = LabelMeanFeaturesRecogV12C.reshape(-1,2,1)  





# path = 'Trained/RawSignalMean1Features_MLP_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'MLP_Mean1_results.txt'),model.predict(InputMeanFeaturesTriclassV1))

# path = 'Trained/RawSignalMean1Features_CNN_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'CNN_Mean1_results.txt'),model.predict(InputMeanFeaturesTriclassV1CNN))

# path = 'Trained/RawSignalMean1Features_LSTM_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Mean1_results.txt'),model.predict(InputMeanFeaturesTriclassV1LSTM))    


path = 'Trained/RawSignalMean5Features_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Mean5_results.txt'),model.predict(InputMeanFeaturesTriclassV5))

path = 'Trained/RawSignalMean5Features_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'CNN_Mean5_results.txt'),model.predict(InputMeanFeaturesTriclassV5CNN))

path = 'Trained/RawSignalMean5Features_LSTM_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Mean5_results.txt'),model.predict(InputMeanFeaturesTriclassV5LSTM))    



path = 'Trained/RawSignalMean12Features_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])  
np.savetxt(pjoin(path,'CNN_Mean12_results.txt'),model.predict(InputMeanFeaturesTriclassV12CNN))

path = 'Trained/RawSignalMean12Features_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Mean12_results.txt'),model.predict(InputMeanFeaturesTriclassV12))

path = 'Trained/RawSignalMean12Features_LSTM_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Mean12_results.txt'),model.predict(InputMeanFeaturesTriclassV12LSTM))





# path = 'Trained/RawSignalMean1Features_MLP_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'MLP_Mean1_results.txt'),model.predict(InputMeanFeaturesRecogV1))

# path = 'Trained/RawSignalMean1Features_CNN_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'CNN_Mean1_results.txt'),model.predict(InputMeanFeaturesRecogV1CNN))

# path = 'Trained/RawSignalMean1Features_LSTM_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Mean1_results.txt'),model.predict(InputMeanFeaturesRecogV1LSTM))   



path = 'Trained/RawSignalMean5Features_MLP_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Mean5_results.txt'),model.predict(InputMeanFeaturesRecogV5))

path = 'Trained/RawSignalMean5Features_CNN_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'CNN_Mean5_results.txt'),model.predict(InputMeanFeaturesRecogV5CNN))

path = 'Trained/RawSignalMean5Features_LSTM_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Mean5_results.txt'),model.predict(InputMeanFeaturesRecogV5LSTM))    



path = 'Trained/RawSignalMean12Features_CNN_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])  
np.savetxt(pjoin(path,'CNN_Mean12_results.txt'),model.predict(InputMeanFeaturesRecogV12CNN))

path = 'Trained/RawSignalMean12Features_MLP_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Mean12_results.txt'),model.predict(InputMeanFeaturesRecogV12))

path = 'Trained/RawSignalMean12Features_LSTM_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Mean12_results.txt'),model.predict(InputMeanFeaturesRecogV12LSTM))












