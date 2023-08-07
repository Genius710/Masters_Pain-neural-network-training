# -*- coding: utf-8 -*-
"""
Created on Tue May 18 18:41:13 2021

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
multi =1
VECL = 68

mat_fname = pjoin('C:\OneDrive - Kaunas University of Technology\MAGISTRINIS\DataMatrixFeatures_ns_4.mat')
mat_contents = sio.loadmat(mat_fname)
DataMatrix = np.array(mat_contents['DataMatrix'])
Input = DataMatrix[:,2:2+multi*VECL]
Label = DataMatrix[:,0]

Input = Input.reshape(-1,VECL*multi,1)



path = 'Trained/RawSignalLong1Features_CNN_v1_triclass/'
model = tf.keras.models.load_model(pjoin(path,'450')) 
model.compile('adam', loss='categorical_crossentropy',metrics=[tf.keras.metrics.CategoricalAccuracy()])  
np.savetxt(pjoin(path,'CNN_Long1_demo4tri.txt'),model.predict(Input))