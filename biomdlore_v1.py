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

multi = 5
VECL = 500
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong05_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLong0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLong15_recog.mat')
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
        
        
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV05_recog.mat')
mat_contents = h5py.File(mat_fname,'r')
DataMatrix0 = np.swapaxes(np.array(mat_contents['EqualisedDataMatrixLongV0']),0,1)
mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\EqualisedDataMatrixLongV15_recog.mat')
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




# multi =1

multi = 5


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

#             tf.keras.layers.Dense(2,activation ='softmax', name="final"),
#         ]
# )


# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi*VECL,1]),

#             tf.keras.layers.Conv1D(100,100 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=5, name="MaxPooling1D"),
#             tf.keras.layers.Conv1D(50,50 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=5),
#             tf.keras.layers.Conv1D(24,24 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=5),
#             tf.keras.layers.Flatten(),

#             tf.keras.layers.Dense(2,activation ='softmax', name="final"),
#         ]
# )

# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi*VECL,1]),

#             tf.keras.layers.Conv1D(5,500 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=5, name="MaxPooling1D"),
#             tf.keras.layers.Conv1D(50,50 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=5),
#             tf.keras.layers.Conv1D(24,24 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=5),
#             tf.keras.layers.Flatten(),

#             tf.keras.layers.Dense(2,activation ='softmax', name="final"),
#         ]
# )


# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi*VECL,1]),

#             tf.keras.layers.Conv1D(5,500 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=5, name="MaxPooling1D"),
#             tf.keras.layers.Conv1D(50,50 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=4),
#             tf.keras.layers.Conv1D(24,24 ,activation='relu'),
#             tf.keras.layers.MaxPooling1D(pool_size=2),
#             tf.keras.layers.Conv1D(24,24 ,activation='relu'),
#             # tf.keras.layers.MaxPooling1D(pool_size=5),
#             # tf.keras.layers.Conv1D(24,24 ,activation='relu'),
#             # tf.keras.layers.MaxPooling1D(pool_size=5),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(256,activation='relu'),
#             tf.keras.layers.Dense(256,activation='relu'),
#             tf.keras.layers.Dense(256,activation='relu'),
#             tf.keras.layers.Dense(256,activation='relu'),
#             tf.keras.layers.Dense(256,activation='relu'),
#             # tf.keras.layers.Dense(256,activation='relu'),
#             # tf.keras.layers.Dense(256,activation='relu'),

#             tf.keras.layers.Dense(2,activation ='softmax', name="final"),
#         ]
# )

model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi*VECL,1]),

            tf.keras.layers.Conv1D(5,500 ,activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=5, name="MaxPooling1D"),
            tf.keras.layers.Conv1D(50,50 ,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling1D(pool_size=4),
            tf.keras.layers.Conv1D(24,24 ,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(24,24 ,activation='relu'),
            tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.MaxPooling1D(pool_size=5),
            # tf.keras.layers.Conv1D(24,24 ,activation='relu'),
            # tf.keras.layers.MaxPooling1D(pool_size=5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256,activation='sigmoid'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256,activation='sigmoid'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256,activation='sigmoid'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256,activation='sigmoid'),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256,activation='sigmoid'),
            # tf.keras.layers.Dropout(0.2),
            
            # tf.keras.layers.Dense(124,activation='sigmoid'),
            # tf.keras.layers.Dense(124,activation='sigmoid'),
            # tf.keras.layers.Dense(124,activation='sigmoid'),
            # tf.keras.layers.Dense(256,activation='relu'),
            # tf.keras.layers.Dense(256,activation='relu'),

            tf.keras.layers.Dense(2,activation ='softmax', name="final"),
        ]
)


eep = 11
epoch_no = 10
size = 256



model.summary()
# model = tf.keras.models.load_model('Trained/RawSignalLong5_CNN_v1_recog/500')
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])

for iii in range(1,eep):
    history = model.fit(InputLong5CNN,LabelLong5CCNN,epochs = epoch_no,batch_size =size,validation_data =(InputLongV5CNN,LabelLongV5CCNN))
    model.save(pjoin('Biomdlore_CNN_v2_recog/',str(iii*epoch_no+0)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])



