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



path = 'Trained/RawSignalMean1_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500'))
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Mean1_results.txt'),model.predict(InputLongTriclassV1))
   
path = 'RawSignalMean1_LSTM_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Mean1_results.txt'),model.predict(InputMeanV1LSTM))   
 
path = 'RawSignalMean1_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'CNN_Mean1_results.txt'),model.predict(InputMeanV1CNN))
    

    
path = 'RawSignalMean5_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Mean5_results.txt'),model.predict(InputMeanV5))

path = 'RawSignalMean5_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'CNN_Mean5_results.txt'),model.predict(InputMeanV5CNN))

path = 'RawSignalMean5_LSTM_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Mean5_results.txt'),model.predict(InputMeanV5LSTM))    



path = 'RawSignalMean12_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'210')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])  
np.savetxt(pjoin(path,'CNN_Mean12_results.txt'),model.predict(InputMeanV12CNN))

path = 'RawSignalMean12_MLP_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'MLP_Mean12_results.txt'),model.predict(InputMeanV12))

path = 'RawSignalMean12_LSTM_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'500')) 
model.compile('adamax', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])
np.savetxt(pjoin(path,'LSTM_Mean12_results.txt'),model.predict(InputMeanV12))


