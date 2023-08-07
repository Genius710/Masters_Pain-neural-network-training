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

mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixMean5Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixMean'])



multi = 1
VECL = 68


Input = DataMatrix[:,2:2+multi*VECL]
Label = DataMatrix[:,0]

DataMatrix =[]     
        

[x] = Label.shape
LabelC = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if Label[iii] ==0:
        LabelC[iii,0] = 1
        LabelC[iii,1] = 0
        LabelC[iii,2] = 0

    elif Label[iii] ==1: 
        LabelC[iii,0] = 0
        LabelC[iii,1] = 1
        LabelC[iii,2] = 0


    elif Label[iii] ==2 : 
        LabelC[iii,0] = 0
        LabelC[iii,1] = 0
        LabelC[iii,2] = 1

        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\\EqualisedDataMatrixMeanV5Features_triclass.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['EqualisedDataMatrixMeanV'])

InputV = DataMatrix[:,2:2+multi*VECL]
LabelV = DataMatrix[:,0]

DataMatrix =[]     

    
[x] = LabelV.shape
LabelVC = np.empty([x,3], dtype=int)
       
for iii in range(0,x):
    
    if LabelV[iii] ==0:
        LabelVC[iii,0] = 1
        LabelVC[iii,1] = 0
        LabelVC[iii,2] = 0
        
    elif LabelV[iii] ==1 : 
        LabelVC[iii,0] = 0
        LabelVC[iii,1] = 1
        LabelVC[iii,2] = 0

    elif LabelV[iii] ==2 : 
        LabelVC[iii,0] = 0
        LabelVC[iii,1] = 0
        LabelVC[iii,2] = 1



# Input = np.transpose(Input)
# InputV = np.transpose(InputV)

# Input = Input.reshape(-1,VECL*multi,1)
# LabelC = LabelC.reshape(-1,3,1)


# InputV = InputV.reshape(-1,VECL*multi,1)
# LabelVC = LabelVC.reshape(-1,3,1)


Input = Input.reshape(-1,multi,VECL)
LabelC = LabelC.reshape(-1,3,1)


InputV = InputV.reshape(-1,multi,VECL)
LabelVC = LabelVC.reshape(-1,3,1)

# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi*VECL,1]),

#             tf.keras.layers.Conv1D(64,multi ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"),
#             tf.keras.layers.Conv1D(128*2,5 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=2),
#             tf.keras.layers.Conv1D(20*2,1 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=2),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(128*2, activation ='relu'),
#             tf.keras.layers.Dense(128*2, activation ='relu'),
#             tf.keras.layers.Dense(64*2, activation ='relu'),
#             tf.keras.layers.Dense(3,activation ='softmax', name="final"),
#         ]
# )

model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi*VECL]),
            
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
            tf.keras.layers.Dense(VECL, activation ='relu'),
 


            tf.keras.layers.Dense(3,activation ='softmax', name="final"),
        ]
)


# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi,VECL]),
            
#             tf.keras.layers.LSTM(VECL*multi),
#             # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
#             # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
#             # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
            
#             tf.keras.layers.Dense(3,activation ='softmax', name="final"),
#         ]
# )

# model = tf.keras.models.load_model('RawSignalLong12_LSTM_v1_triclass/200')

# model = tf.keras.models.load_model('Feature_model_53_CNN_aveg_v1/200')


model.summary()

# model.compile('adam', loss='mse')
# model.compile('sgd', loss='binary_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
#return model.summary()
# history = model.fit(Input,LabelF,epochs = 20,)

for iii in range(1,2):
    history = model.fit(Input,LabelC,epochs = 1,batch_size =32,validation_data =(InputV,LabelVC))
    model.save(pjoin('RawSignalMean5Features_LSTM_v1_triclass/',str(1)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
    

# plt.plot(np.argmax(model.predict(InputV),1))
# plt.plot(LabelV)








