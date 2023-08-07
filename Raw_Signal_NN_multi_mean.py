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
VECL = 200






mat_fname = pjoin(r"C:\OneDrive - Kaunas University of Technology\Nivelos projektas\Datamatrix.mat")
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.swapaxes(np.array(mat_contents['DataMatrix']),0,1)

Input =  DataMatrix[1:201,:]
Label = DataMatrix[0,:]



[x] = Label.shape
LabelC = np.empty([x,4], dtype=int)
       
for iii in range(0,x):
    
    if Label[iii] ==0:
        LabelC[iii,0] = 1
        LabelC[iii,1] = 0
        LabelC[iii,2] = 0
        LabelC[iii,3] = 0

    elif Label[iii] ==1: 
        LabelC[iii,0] = 0
        LabelC[iii,1] = 1
        LabelC[iii,2] = 0
        LabelC[iii,3] = 0


    elif Label[iii] ==2 : 
        LabelC[iii,0] = 0
        LabelC[iii,1] = 0
        LabelC[iii,2] = 1
        LabelC[iii,3] = 0
    elif Label[iii] ==3 : 
        LabelC[iii,0] = 0
        LabelC[iii,1] = 0
        LabelC[iii,2] = 0
        LabelC[iii,3] = 1
        
        



Input = Input.reshape(-1,VECL*multi,1)
LabelC = LabelC.reshape(-1,4,1)



model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=[multi*VECL,1]),

            tf.keras.layers.Conv1D(64,multi ,activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"),
            tf.keras.layers.Conv1D(128*2,5 ,activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(20*2,1 ,activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128*2, activation ='relu'),
            tf.keras.layers.Dense(128*2, activation ='relu'),
            tf.keras.layers.Dense(64*2, activation ='relu'),
            tf.keras.layers.Dense(4,activation ='softmax', name="final"),
        ]
)

# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi*VECL]),
            
#             tf.keras.layers.Dense(VECL, activation ='relu'),
#             tf.keras.layers.Dense(VECL, activation ='relu'),
#             tf.keras.layers.Dense(VECL, activation ='relu'),
#             tf.keras.layers.Dense(VECL, activation ='relu'),
 


#             tf.keras.layers.Dense(4,activation ='softmax', name="final"),
#         ]
# )


# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Input(shape=[multi,VECL]),
            
#             tf.keras.layers.LSTM(VECL*multi),
#             # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
#             # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
#             # tf.keras.layers.Dense(VACL*multi, activation ='relu'),
            
#             tf.keras.layers.Dense(4,activation ='softmax', name="final"),
#         ]
# )

# model = tf.keras.models.load_model('Feature_model_53_v2/1900')

# model = tf.keras.models.load_model('Feature_model_53_CNN_aveg_v1/200')


model.summary()

# model.compile('adam', loss='mse')
# model.compile('sgd', loss='binary_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
#return model.summary()
# history = model.fit(Input,LabelF,epochs = 20,)

for iii in range(1,50):
    history = model.fit(Input,LabelC,epochs = 10,batch_size =256)
    model.save(pjoin('MLP_v1/',str(iii*10)))
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['categorical_accuracy'])
    

# plt.plot(np.argmax(model.predict(InputV),1))
# plt.plot(LabelV)








