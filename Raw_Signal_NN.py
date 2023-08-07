# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 02:49:00 2020

@author: Povilas-Predator-PC
"""

from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np

import tensorflow as tf

import keras

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixEqualized_v2.mat')
# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixEqualized_53_multi_12_averaged_v2.mat')
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMean5.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=1)
DataMatrix = mat_contents['DataMatrixMean']
# DataMatrix = DataMatrix[0,:]

multi = 1
VECL = 500

Input = DataMatrix[:,2:2+multi*VECL]
Label = DataMatrix[:,0]

# Input = np.empty([1,53], dtype=float)
# Label = np.empty([1], dtype=int)

# for iii in range(0,50):
#     if (iii ==2 or iii ==20): 
#         print(iii)
#     else:
#         Input =  np.vstack(( Input,DataMatrix[iii][:,2:55]))
#         Label = np.concatenate(( Label,DataMatrix[iii][:,0]))
        
        

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

mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixMeanV5.mat')
# mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixValidationSelected_multi_12_averaged_v2.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=1)
DataMatrix = mat_contents['DataMatrixMeanV']
# DataMatrix = DataMatrix[0,:]

InputV = DataMatrix[:,2:2+multi*VECL]
LabelV = DataMatrix[:,0]


# InputV = np.empty([1,53], dtype=float)
# LabelV = np.empty([1], dtype=int)

# for iii in range(50,51):
#     if (iii ==2 or iii ==20): 
#         print(iii)
#     else:
#         InputV =  np.vstack(( InputV,DataMatrix[iii][:,2:55]))
#         LabelV = np.concatenate(( LabelV,DataMatrix[iii][:,0]))
        
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


# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.LSTM(128,batch_input_shape = (None,53,1),return_sequences = 0),
#             tf.keras.layers.Dense(3,activation='sigmoid', name="final"),
#         ]
# )

# Input = Input.reshape(-1,VECL*multi,1)
# LabelC = LabelC.reshape(-1,3,1)


# InputV = InputV.reshape(-1,VECL*multi,1)
# LabelVC = LabelVC.reshape(-1,3,1)

# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi*VECL,1]),

#             tf.keras.layers.Conv1D(53*2,10 ,activation='relu'),
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
            
            tf.keras.layers.Dense(1500, activation ='relu'),
            # tf.keras.layers.Dense(1500, activation ='relu'),
            # tf.keras.layers.Dense(1500, activation ='relu'),
            
            tf.keras.layers.Dense(600, activation ='relu'),
            # tf.keras.layers.Dense(600, activation ='relu'),
            # tf.keras.layers.Dense(600, activation ='relu'),
            
            tf.keras.layers.Dense(300, activation ='relu'),
            # tf.keras.layers.Dense(300, activation ='relu'),
            # tf.keras.layers.Dense(300, activation ='relu'),
            
            tf.keras.layers.Dense(300, activation ='relu'),
            # tf.keras.layers.Dense(300, activation ='relu'),
            # tf.keras.layers.Dense(300, activation ='relu'),
            # tf.keras.layers.Dense(164, activation ='relu'),
            # tf.keras.layers.Dense(164, activation ='relu'),
            
            # tf.keras.layers.Dense(164, activation ='relu'),
            # tf.keras.layers.Dense(164, activation ='relu'),
            # tf.keras.layers.Dense(164, activation ='relu'),
            
            # tf.keras.layers.Dense(164, activation ='relu'),
            # tf.keras.layers.Dense(164, activation ='relu'),
            # tf.keras.layers.Dense(164, activation ='relu'),


            tf.keras.layers.Dense(3,activation ='softmax', name="final"),
        ]
)

# model = tf.keras.models.load_model('Feature_model_53_v2/1900')

# model = tf.keras.models.load_model('Feature_model_53_CNN_aveg_v1/200')


model.summary()

# model.compile('adam', loss='mse')
# model.compile('sgd', loss='binary_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
#return model.summary()
# history = model.fit(Input,LabelF,epochs = 20,)

for iii in range(1,10):
    history = model.fit(Input,LabelC,epochs = 100,batch_size =100,validation_data =(InputV,LabelVC))
    model.save(pjoin('RawSignalMean5_MLP_v1/',str(iii*100)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
    

# plt.plot(np.argmax(model.predict(InputV),1))
# plt.plot(LabelV)








