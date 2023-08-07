# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:45:21 2020

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

# model = tf.keras.models.load_model('MLP_MBC_CNAP_FH_Train_cat_12_3_4_56/100')
model = tf.keras.models.load_model('MLB12_MB_PPG_FH_Train_cat_12_3_4_56/500')


# mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Dataset/PPG/SBC_test50.mat')
mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Dataset/PPG/NE_MBC_test.mat')

# mat_fname = pjoin('D:\OneDrive - Kaunas University of Technology\MAGISTRINIS\Data\Aurimod50CNAPBlockV2.mat')

mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
Input = mat_contents['Input_train']
Label = mat_contents['Label']
# Input = mat_contents['Input']


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


model.summary()
model.compile('adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.evaluate(Input,LabelF)


x = Label.size


result = np.empty([x,5], dtype=float)
resultN = np.empty(x, dtype=float)
LabelN = np.empty(x, dtype=float)
print('prediction started')
result = model.predict(Input)

for iii in range(0,x):
    resultN[iii] = np.argmax(result[iii,:])
    

for iii in range(0,x):
    LabelN[iii] = np.argmax(LabelF[iii,:])
    
plt.plot(resultN)  
plt.plot(LabelN+0.25)


