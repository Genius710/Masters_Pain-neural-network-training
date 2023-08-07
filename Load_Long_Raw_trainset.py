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


# def Load_L13():
multi = 1
VECL = 500


mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLong1_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLong'])
InputLongTriclassV1 = DataMatrix[:,2:2+multi*VECL]
LabelLongTriclassV1 = DataMatrix[:,0]

DataMatrix =[]     
        
        

[x] = LabelLongTriclassV1.shape
LabelLongTriclassV1C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongTriclassV1[iii] ==0:
        LabelLongTriclassV1C[iii,0] = 1
        LabelLongTriclassV1C[iii,1] = 0
        LabelLongTriclassV1C[iii,2] = 0


    elif LabelLongTriclassV1[iii] ==1: 
        LabelLongTriclassV1C[iii,0] = 0
        LabelLongTriclassV1C[iii,1] = 1
        LabelLongTriclassV1C[iii,2] = 0
        
    elif LabelLongTriclassV1[iii] ==2: 
        LabelLongTriclassV1C[iii,0] = 0
        LabelLongTriclassV1C[iii,1] = 0
        LabelLongTriclassV1C[iii,2] = 1
        

InputLongTriclassV1CNN = InputLongTriclassV1.reshape(-1,VECL*multi,1)
LabelLongTriclassV1CCNN = LabelLongTriclassV1C.reshape(-1,3,1)

InputLongTriclassV1LSTM = InputLongTriclassV1.reshape(-1,multi,VECL)
LabelLongTriclassV1CLSTM = LabelLongTriclassV1C.reshape(-1,3,1)        
        
# return  InputLongTriclassV1, LabelLongTriclassV1C,InputLongTriclassV1CNN, LabelLongTriclassV1CCNN,InputLongTriclassV1LSTM, LabelLongTriclassV1CLSTM
   # 
# def Load_L53():        
multi =5     
VECL = 500
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLong5_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLong'])
InputLongTriclassV5 = DataMatrix[:,2:2+multi*VECL]
LabelLongTriclassV5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLongTriclassV5.shape
LabelLongTriclassV5C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongTriclassV5[iii] ==0:
        LabelLongTriclassV5C[iii,0] = 1
        LabelLongTriclassV5C[iii,1] = 0
        LabelLongTriclassV5C[iii,2] = 0


    elif LabelLongTriclassV5[iii] ==1: 
        LabelLongTriclassV5C[iii,0] = 0
        LabelLongTriclassV5C[iii,1] = 1
        LabelLongTriclassV5C[iii,2] = 0
    elif LabelLongTriclassV5[iii] ==2: 
        LabelLongTriclassV5C[iii,0] = 0
        LabelLongTriclassV5C[iii,1] = 0
        LabelLongTriclassV5C[iii,2] = 1
        


InputLongTriclassV5CNN = InputLongTriclassV5.reshape(-1,VECL*multi,1)
LabelLongTriclassV5CCNN = LabelLongTriclassV5C.reshape(-1,3,1)


InputLongTriclassV5LSTM = InputLongTriclassV5.reshape(-1,multi,VECL)
LabelLongTriclassV5CLSTM = LabelLongTriclassV5C.reshape(-1,3,1)                

# return  InputLongTriclassV5, LabelLongTriclassV5C,InputLongTriclassV5CNN, LabelLongTriclassV5CCNN,InputLongTriclassV5LSTM, LabelLongTriclassV5CLSTM
   
# def Load_L123():     
multi =12     
VECL = 500
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLong12_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLong'])
InputLongTriclassV12 = DataMatrix[:,2:2+multi*VECL]
LabelLongTriclassV12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLongTriclassV12.shape
LabelLongTriclassV12C = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongTriclassV12[iii] ==0:
        LabelLongTriclassV12C[iii,0] = 1
        LabelLongTriclassV12C[iii,1] = 0
        LabelLongTriclassV12C[iii,2] = 0


    elif LabelLongTriclassV12[iii] ==1: 
        LabelLongTriclassV12C[iii,0] = 0
        LabelLongTriclassV12C[iii,1] = 1
        LabelLongTriclassV12C[iii,2] = 0
    elif LabelLongTriclassV12[iii] ==2: 
        LabelLongTriclassV12C[iii,0] = 0
        LabelLongTriclassV12C[iii,1] = 0
        LabelLongTriclassV12C[iii,2] = 1


InputLongTriclassV12CNN = InputLongTriclassV12.reshape(-1,VECL*multi,1)
LabelLongTriclassV12CCNN = LabelLongTriclassV12C.reshape(-1,3,1)


InputLongTriclassV12LSTM = InputLongTriclassV12.reshape(-1,multi,VECL)
LabelLongTriclassV12CLSTM = LabelLongTriclassV12C.reshape(-1,3,1)  

# return  InputLongTriclassV12, LabelLongTriclassV12C,InputLongTriclassV12CNN, LabelLongTriclassV12CCNN,InputLongTriclassV12LSTM, LabelLongTriclassV12CLSTM



# def Load_L12():    

multi = 1
VECL = 500


mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLong1_recog.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLong'])
InputLongRecogV1 = DataMatrix[:,2:2+multi*VECL]
LabelLongRecogV1 = DataMatrix[:,0]

DataMatrix =[]     
        
        

[x] = LabelLongRecogV1.shape
LabelLongRecogV1C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongRecogV1[iii] ==0:
        LabelLongRecogV1C[iii,0] = 1
        LabelLongRecogV1C[iii,1] = 0


    elif LabelLongRecogV1[iii] ==1: 
        LabelLongRecogV1C[iii,0] = 0
        LabelLongRecogV1C[iii,1] = 1

        

InputLongRecogV1CNN = InputLongRecogV1.reshape(-1,VECL*multi,1)
LabelLongRecogV1CCNN = LabelLongRecogV1C.reshape(-1,2,1)

InputLongRecogV1LSTM = InputLongRecogV1.reshape(-1,multi,VECL)
LabelLongRecogV1CLSTM = LabelLongRecogV1C.reshape(-1,2,1)

# return  InputLongRecogV1, LabelLongRecogV1C,InputLongRecogV1CNN, LabelLongRecogV1CCNN,InputLongRecogV1LSTM, LabelLongRecogV1CLSTM
    
        
# def Load_L52():    


VECL = 500       
multi =5        
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLong5_recog.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrixLong'])
InputLongRecogV5 = DataMatrix[:,2:2+multi*VECL]
LabelLongRecogV5 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLongRecogV5.shape
LabelLongRecogV5C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongRecogV5[iii] ==0:
        LabelLongRecogV5C[iii,0] = 1
        LabelLongRecogV5C[iii,1] = 0


    elif LabelLongRecogV5[iii] ==1: 
        LabelLongRecogV5C[iii,0] = 0
        LabelLongRecogV5C[iii,1] = 1

        


InputLongRecogV5CNN = InputLongRecogV5.reshape(-1,VECL*multi,1)
LabelLongRecogV5CCNN = LabelLongRecogV5C.reshape(-1,2,1)


InputLongRecogV5LSTM = InputLongRecogV5.reshape(-1,multi,VECL)
LabelLongRecogV5CLSTM = LabelLongRecogV5C.reshape(-1,2,1)                

#     return  InputLongRecogV5, LabelLongRecogV5C,InputLongRecogV5CNN, LabelLongRecogV5CCNN,InputLongRecogV5LSTM, LabelLongRecogV5CLSTM

# def Load_L122():  

multi =12        
VECL = 500 



        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixLong12_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix = np.swapaxes(np.array(mat_contents['DataMatrixLong']),0,1)
InputLongRecogV12 = DataMatrix[:,2:2+multi*VECL]
LabelLongRecogV12 = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = LabelLongRecogV12.shape
LabelLongRecogV12C = np.empty([x,2], dtype=int)
       
for iii in range(0,x):
    
    if LabelLongRecogV12[iii] ==0:
        LabelLongRecogV12C[iii,0] = 1
        LabelLongRecogV12C[iii,1] = 0



    elif LabelLongRecogV12[iii] ==1: 
        LabelLongRecogV12C[iii,0] = 0
        LabelLongRecogV12C[iii,1] = 1



InputLongRecogV12CNN = InputLongRecogV12.reshape(-1,VECL*multi,1)
LabelLongRecogV12CCNN = LabelLongRecogV12C.reshape(-1,2,1)


InputLongRecogV12LSTM = InputLongRecogV12.reshape(-1,multi,VECL)
LabelLongRecogV12CLSTM = LabelLongRecogV12C.reshape(-1,2,1)  

# return  InputLongRecogV12, LabelLongRecogV12C,InputLongRecogV12CNN, LabelLongRecogV12CCNN,InputLongRecogV12LSTM, LabelLongRecogV12CLSTM



path = 'Trained/RawSignalLong1_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Long1_train_results.txt'),model.predict(InputLongTriclassV1))

path = 'Trained/RawSignalLong1_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'CNN_Long1_train_results.txt'),model.predict(InputLongTriclassV1CNN))

path = 'Trained/RawSignalLong1_LSTM_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Long1_train_results.txt'),model.predict(InputLongTriclassV1LSTM))    


path = 'Trained/RawSignalLong5_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Long5_train_results.txt'),model.predict(InputLongTriclassV5))

path = 'Trained/RawSignalLong5_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'CNN_Long5_train_results.txt'),model.predict(InputLongTriclassV5CNN))

path = 'Trained/RawSignalLong5_LSTM_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Long5_train_results.txt'),model.predict(InputLongTriclassV5LSTM))    



path = 'Trained/RawSignalLong12_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])  
np.savetxt(pjoin(path,'CNN_Long12_train_results.txt'),model.predict(InputLongTriclassV12CNN))

path = 'Trained/RawSignalLong12_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Long12_train_results.txt'),model.predict(InputLongTriclassV12))





# path = 'Trained/RawSignalLong12_LSTM_v1_triclass/'
# model = tf.keras.models.load_model(pjoin(path,'450')) 
# model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Long12_train_results.txt'),model.predict(InputLongTriclassV12LSTM))





path = 'Trained/RawSignalLong1_MLP_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Long1_train_results.txt'),model.predict(InputLongRecogV1))

path = 'Trained/RawSignalLong1_CNN_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'CNN_Long1_train_results.txt'),model.predict(InputLongRecogV1CNN))

path = 'Trained/RawSignalLong1_LSTM_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Long1_train_results.txt'),model.predict(InputLongRecogV1LSTM))   



path = 'Trained/RawSignalLong5_MLP_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Long5_train_results.txt'),model.predict(InputLongRecogV5))

path = 'Trained/RawSignalLong5_CNN_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'CNN_Long5_train_results.txt'),model.predict(InputLongRecogV5CNN))

path = 'Trained/RawSignalLong5_LSTM_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'300')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Long5_train_results.txt'),model.predict(InputLongRecogV5LSTM))    



path = 'Trained/RawSignalLong12_CNN_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'4')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])  
np.savetxt(pjoin(path,'CNN_Long12_train_results.txt'),model.predict(InputLongRecogV12CNN))

path = 'Trained/RawSignalLong12_MLP_v1_recog/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Long12_train_results.txt'),model.predict(InputLongRecogV12))
model.evaluate(InputLongRecogV12,LabelLongRecogV12C)

# path = 'Trained/RawSignalLong12_LSTM_v1_recog/'
# model = tf.keras.models.load_model(pjoin(path,'500')) 
# model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
# np.savetxt(pjoin(path,'LSTM_Long12_train_results.txt'),model.predict(InputLongRecogV12LSTM))







