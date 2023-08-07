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

mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Dataset/PPG/FH_MB_test.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
Input = mat_contents['Input_train']
Label = mat_contents['Label']

mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Dataset/PPG/MB_test50.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
InputV = mat_contents['Input_train']
LabelV = mat_contents['Label']

[x,y] = LabelV.shape
LabelVF = np.empty([x,4], dtype=int)
       
for iii in range(0,x):
    
    if LabelV[iii] ==0:
        LabelVF[iii,0] = 1
        LabelVF[iii,1] = 0
        LabelVF[iii,2] = 0
        LabelVF[iii,3] = 0
        # LabelVF[iii,4] = 0
        # LabelVF[iii,5] = 0
    elif LabelV[iii] ==1: 
        LabelVF[iii,0] = 1
        LabelVF[iii,1] = 0
        LabelVF[iii,2] = 0
        LabelVF[iii,3] = 0
        # LabelVF[iii,4] = 0
        # LabelVF[iii,5] = 0
    elif LabelV[iii] ==2:
        LabelVF[iii,0] = 0
        LabelVF[iii,1] = 1
        LabelVF[iii,2] = 0
        LabelVF[iii,3] = 0
        # LabelVF[iii,4] = 0
        # LabelVF[iii,5] = 0
    elif LabelV[iii] ==3:
        LabelVF[iii,0] = 0
        LabelVF[iii,1] = 0
        LabelVF[iii,2] = 1
        LabelVF[iii,3] = 0
        # LabelVF[iii,4] = 0
        # LabelVF[iii,5] = 0
    elif LabelV[iii] >= 4:
        LabelVF[iii,0] = 0
        LabelVF[iii,1] = 0
        LabelVF[iii,2] = 0
        LabelVF[iii,3] = 1
        # LabelVF[iii,4] = 1
        # LabelVF[iii,5] = 0

# InputV = InputV.reshape(-1,100,1)
# LabelVF = LabelVF.reshape(-1,4,1)

[x,y] = Label.shape
LabelF = np.empty([x,4], dtype=int)
       
for iii in range(0,x):
    
    if Label[iii] ==0:
        LabelF[iii,0] = 1
        LabelF[iii,1] = 0
        LabelF[iii,2] = 0
        LabelF[iii,3] = 0
        # LabelF[iii,4] = 0
        # LabelF[iii,5] = 0
    elif Label[iii] ==1: 
        LabelF[iii,0] = 1
        LabelF[iii,1] = 0
        LabelF[iii,2] = 0
        LabelF[iii,3] = 0
        # LabelF[iii,4] = 0
        # LabelF[iii,5] = 0
    elif Label[iii] ==2:
        LabelF[iii,0] = 0
        LabelF[iii,1] = 1
        LabelF[iii,2] = 0
        LabelF[iii,3] = 0
        # LabelF[iii,4] = 0
        # LabelF[iii,5] = 0
    elif Label[iii] ==3:
        LabelF[iii,0] = 0
        LabelF[iii,1] = 0
        LabelF[iii,2] = 1
        LabelF[iii,3] = 0
        # LabelF[iii,4] = 0
        # LabelF[iii,5] = 0
    elif Label[iii] >= 4:
        LabelF[iii,0] = 0
        LabelF[iii,1] = 0
        LabelF[iii,2] = 0
        LabelF[iii,3] = 1
        # LabelF[iii,4] = 1
        # LabelF[iii,5] = 0

# Input = Input.reshape(-1,100,1)
# LabelF = LabelF.reshape(-1,4,1)


# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.LSTM(128,batch_input_shape = (None,100,1),return_sequences = 0),
#             tf.keras.layers.Dense(4,activation='sigmoid', name="final"),
#         ]
# )

# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Dense(300, input_shape=[500]),
#             tf.keras.layers.Dense(300, activation = 'sigmoid'),
#             tf.keras.layers.Dense(300, activation = 'sigmoid'),
#             tf.keras.layers.Dense(300, activation = 'sigmoid'),
#             tf.keras.layers.Dense(300, activation = 'sigmoid'),
            

#             tf.keras.layers.Dense(300, activation = 'relu'),
#             tf.keras.layers.Dense(300, activation = 'relu'),
#             tf.keras.layers.Dense(300, activation = 'relu'),
#             tf.keras.layers.Dense(300, activation = 'relu'),
            
#             tf.keras.layers.Dense(300, activation = 'sigmoid'),
#             tf.keras.layers.Dense(300, activation = 'sigmoid'),
#             tf.keras.layers.Dense(300, activation = 'sigmoid'),
#             tf.keras.layers.Dense(300, activation = 'sigmoid'),
            
#             # tf.keras.layers.Dense(3000, activation = 'swish'),
#             # tf.keras.layers.Dense(300, activation = 'sigmoid'),
#             # tf.keras.layers.Dense(300, activation = 'swish'),
#             # tf.keras.layers.Dense(3000, activation = 'sigmoid'),
#             # tf.keras.layers.Dense(3000, activation = 'sigmoid'),
#             # tf.keras.layers.Dense(3000, activation = 'sigmoid'),
#             tf.keras.layers.Dense(4,activation ='softmax', name="final"),
#         ]
# )

model = tf.keras.models.load_model('MLB12_MB_CNAP_FH_Train_cat_12_3_4_56/1100')
# model = tf.keras.models.load_model('MLB12_SB_PPG_FH_Train_cat_12_3_4_56/400')

# opt = tf.keras.optimizers.SGD(
#     learning_rate=0.0001, momentum=0.0, nesterov=False, name="SGD")

model.summary()

# model.compile('adam', loss='mse')
# model.compile('sgd', loss='binary_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.compile('adamax', loss='binary_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
#return model.summary()
# history = model.fit(Input,LabelF,epochs = 20,)

for iii in range(1,6):
    history = model.fit(Input,LabelF,epochs = 100,batch_size =6000,validation_data =(InputV,LabelVF))
    plt.plot(history.history['categorical_accuracy'])
    model.save(pjoin('MLB12_MB_PPG_FH_Train_cat_12_3_4_56/',str(iii*100)))









