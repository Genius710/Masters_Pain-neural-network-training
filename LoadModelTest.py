# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:01:40 2020

@author: Povilas-Predator-PC
"""
from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np

import tensorflow as tf

import keras
model = tf.keras.models.load_model('lstm2D_Epoch/1160')
num = str(1)

mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'CNAPBlock.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
InputN = mat_contents['Input']
LabelN = mat_contents['Label']
NPRSN = mat_contents['NPRS']

# blacklist = [22,24,35,38]
# whitelist = [25,27,28,29,41,46,48]
# soso = [23,26,30,31,32,33,34,36,37,40,43,44,45,49,50,51]
# whitelist =51
# size = 22
whitelist = [23,25,26,27,28,29,30,31,32,34,35,36,37,38,40,41,43,44,45,46,48,49,50,51]
size = len(whitelist)+21
for iii in range(2,22):
    num = str(iii)
    if iii <10:
        print('Aurimod0' +num +'CNAPBlock.mat')
        mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'CNAPBlock.mat')
    else:
        print('Aurimod' +num +'CNAPBlock.mat')
        mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod' +num +'CNAPBlock.mat') 
    mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
    Input = mat_contents['Input']
    Label = mat_contents['Label']
    NPRS = mat_contents['NPRS']
    InputN = np.vstack((InputN,Input))
    LabelN = np.vstack((LabelN,Label))
    NPRSN = np.hstack((NPRSN,NPRS))


for iii in range(21,size):
    num = str(whitelist[iii-21])
    # num = str(whitelist)
    if iii <10:
        print('Aurimod0' +num +'CNAPBlock.mat')
        mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'CNAPBlock.mat')
    else:
        print('Aurimod' +num +'CNAPBlock.mat')
        mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod' +num +'CNAPBlock.mat') 
    mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
    Input = mat_contents['Input']
    Label = mat_contents['Label']
    NPRS = mat_contents['NPRS']
    InputN = np.vstack((InputN,Input))
    LabelN = np.vstack((LabelN,Label))
    NPRSN = np.hstack((NPRSN,NPRS))
    


[x,y] = NPRSN.shape
NPRSNF = np.empty([y,5], dtype=int)
var = np.empty(1, dtype=int)
NPRSN =NPRSN[0]
for iii in range(0,y):
    # var = np.round(NPRSN[iii]/10)
    # var = var.astype(int)
    # if var ==0:
    #     NPRSNF[iii,:] =np.hstack((np.ones(1),np.zeros(9)))
    # else:
    #     NPRSNF[iii,:] =np.hstack(
    #         (
    #             np.zeros(var-1),
    #             np.ones(1),
    #             np.zeros(10-var)
                
    #             )
    #         )
    if NPRSN[iii] ==0:
        NPRSNF[iii,0] = 1
        NPRSNF[iii,1] = 0
        NPRSNF[iii,2] = 0
        NPRSNF[iii,3] = 0
        NPRSNF[iii,4] = 0
    elif NPRSN[iii] > 0 and NPRSN[iii] <25: 
        NPRSNF[iii,0] = 0
        NPRSNF[iii,1] = 1
        NPRSNF[iii,2] = 0
        NPRSNF[iii,3] = 0
        NPRSNF[iii,4] = 0
    elif NPRSN[iii] >= 25 and NPRSN[iii] <50:
        NPRSNF[iii,0] = 0
        NPRSNF[iii,1] = 0
        NPRSNF[iii,2] = 1
        NPRSNF[iii,3] = 0
        NPRSNF[iii,4] = 0
    elif NPRSN[iii] >= 50 and NPRSN[iii] <75:
        NPRSNF[iii,0] = 0
        NPRSNF[iii,1] = 0
        NPRSNF[iii,2] = 0
        NPRSNF[iii,3] = 1
        NPRSNF[iii,4] = 0
    elif NPRSN[iii] >= 75:
        NPRSNF[iii,0] = 0
        NPRSNF[iii,1] = 0
        NPRSNF[iii,2] = 0
        NPRSNF[iii,3] = 0
        NPRSNF[iii,4] = 1
    
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# InputN = InputN.reshape(-1,104,1)
LabelN = LabelN.reshape(-1,101,1)

NPRSNF = NPRSNF.reshape(-1,5,1)

en = InputN.shape
en = max(en)
# InputF = np.empty([en,300], dtype=float)        
# for iii in range(2,en-1):
#     InputF[iii,:] = np.hstack((InputN[iii,0:100],
#                        InputN[iii-1,0:100],
#                        InputN[iii-2,0:100]
#                        ))
    
# InputF = InputF.reshape(-1,300,1)

# InputF = np.empty([en,20], dtype=float)        
# for iii in range(4,en-1):
#     InputF[iii,:] = np.hstack((InputN[iii-4,100:104],
#                        InputN[iii-3,100:104],
#                        InputN[iii-2,100:104],
#                        InputN[iii-1,100:104],
#                        InputN[iii,100:104],
#                        ))
    
# InputF = InputF.reshape(-1,520,1)


InputF = np.empty([en,100,10], dtype=float)        
for iii in range(9,en-1):
    for yyy in range(1,10):
        InputF[iii,0:100,yyy] = InputN[iii-10+yyy,0:100]

x_train,  y_train,x_test, y_test = train_test_split(InputF, NPRSNF, test_size=0.5)

# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

opt = tf.keras.optimizers.SGD(
    learning_rate=0.00001, momentum=0.0, nesterov=False, name="SGD")



# model.compile(opt, loss='mean_squared_error')
model.compile('adam', loss='binary_crossentropy')
model.summary()
for iii in range(0,300):
    history = model.fit(x_train,x_test,epochs = 10,validation_data =(y_train,y_test))
    plt.plot(history.history['loss'])
    model.save(pjoin('lstm2D_PPG_Epoch/',str(iii*10)))






x = max(InputF.shape)
result = np.empty([x,5], dtype=float)
resultN = np.empty([x], dtype=float)
result = model.predict(InputF)
    
    
for iii in range(0,x):
    resultN[iii] = np.argmax(result[iii,:])
    
plt.figure(dpi=300)
plt.plot(resultN)  
plt.plot(NPRSN)




















