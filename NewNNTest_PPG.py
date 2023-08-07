# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:53:05 2020

@author: Povilas-Predator-PC
"""

from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
num = str(1)

mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'PPGBlockECGXC_Lim1.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
InputN = mat_contents['Input']
LabelN = mat_contents['Label']
NPRSN = mat_contents['NPRS']

blacklist = [ 3, 6, 8, 9, 11, 15, 18, 21, 29, 31, 33, 34, 35, 39, 40, 42, 44, 46, 47, 49 ]
len(blacklist)

for iii in range(1,52):
    if iii in blacklist:
        print('blacklisted')
    else:
        print(iii)
        num = str(iii)
        # num = str(whitelist)
        if iii <10:
            print('Aurimod0' +num +'PPGBlockECGXC_Lim1.mat')
            mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'PPGBlockECGXC_Lim1.mat')
        else:
            print('Aurimod' +num +'PPGBlockECGXC_Lim1.mat')
            mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod' +num +'PPGBlockECGXC_Lim1.mat') 
        mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
        Input = mat_contents['Input']
        Label = mat_contents['Label']
        NPRS = mat_contents['NPRS']
        InputN = np.vstack((InputN,Input))
        LabelN = np.vstack((LabelN,Label))
        NPRSN = np.hstack((NPRSN,NPRS))
    

[x,y] = NPRSN.shape
NPRSNF = np.empty([y,10], dtype=int)
var = np.empty(1, dtype=int)
NPRSN =NPRSN[0]
# for iii in range(0,y):
#     var = np.round(NPRSN[iii]/10)
#     var = var.astype(int)
#     if var ==0:
#         NPRSNF[iii,:] =np.hstack((np.ones(1),np.zeros(9)))
#     else:
#         NPRSNF[iii,:] =np.hstack(
#             (
#                 np.zeros(var-1),
#                 np.ones(1),
#                 np.zeros(10-var)
                
#                 )
#             )
      
    

NPRSNF = np.empty([y,5], dtype=int)

for iii in range(0,y):
    
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
        
en = InputN.shape
en = max(en)
# InputF = np.empty([en,500], dtype=float)        
# for iii in range(4,en-1):
#     InputF[iii,:] = np.hstack((InputN[iii,0:100],
#                        InputN[iii-1,0:100],
#                        InputN[iii-2,0:100],
#                        InputN[iii-3,0:100],
#                        InputN[iii-4,0:100],
#                        ))

# InputF = np.empty([en,20], dtype=float)        
# for iii in range(4,en-1):
#     InputF[iii,:] = np.hstack((InputN[iii-4,100:104],
#                        InputN[iii-3,100:104],
#                        InputN[iii-2,100:104],
#                        InputN[iii-1,100:104],
#                        InputN[iii,100:104],
#                        ))
    
# InputF = np.empty([en,40], dtype=float)        
# for iii in range(9,en-1):
#     InputF[iii,0:4] = InputN[iii-10,100:104]
#     for yyy in range(1,10):
#         InputF[iii,0:(yyy+1)*4] = np.hstack((
#             InputF[iii,0:(yyy)*4],InputN[iii-10+yyy,100:104]
#                            ))
    
# InputF = InputF.reshape(-1,40,1)


# InputF = np.empty([en,1000], dtype=float)        
# for iii in range(9,en-1):
#     InputF[iii,0:100] = InputN[iii-10,0:100]
#     for yyy in range(1,10):
#         InputF[iii,0:(yyy+1)*100] = np.hstack((
#             InputF[iii,0:(yyy)*100],InputN[iii-10+yyy,0:100]
#                            ))
    
# InputF = InputF.reshape(-1,1000,1)
    

InputF = np.empty([en,100,10], dtype=float)        
for iii in range(9,en-1):
    for yyy in range(1,10):
        InputF[iii,0:100,yyy] = InputN[iii-10+yyy,0:100]
                           
        
        
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# InputN = InputN.reshape(-1,104,1)

# LabelN = LabelN.reshape(-1,101,1)

NPRSNF = NPRSNF.reshape(-1,5,1)

x_train,  y_train,x_test, y_test = train_test_split(InputF, NPRSNF, test_size=0.2)
# InputV = InputV.reshape(-1,104,1)
# LabelV = LabelV.reshape(-1,101,1)


# NPRSVF = NPRSVF.reshape(-1,5,1)




import tensorflow as tf
import keras



#def mod(y_new):
    
# model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Dense(104, input_shape=[104,1], name="layer1"),
#             tf.keras.layers.LSTM(100),
#             tf.keras.layers.Dense(101,activation='softmax', name="layer3"),
#         ]
# )


model = tf.keras.Sequential(
        [
            # tf.keras.layers.Dense(300, input_shape=[20],activation = 'sigmoid', name="layer1"),
            # tf.keras.layers.Conv1D(75,20,activation = 'relu' ),
            #  tf.keras.layers.Conv1D(48,20,activation = 'relu' ),
            # # tf.keras.layers.Dropout(0.5),
            
            
        
            tf.keras.layers.GRU(128*2,batch_input_shape = (128,100,10),return_sequences = 0),
            # tf.keras.layers.LSTM(128,return_sequences = False),
            # tf.keras.layers.LSTM(1,batch_input_shape = (None,104,1),return_sequences = True),
            # tf.keras.layers.LSTM(128,return_sequences = True),
            # tf.keras.layers.LSTM(128,return_sequences = False),
            
            
            
            # tf.keras.layers.Dropout(0.25),
            # tf.keras.layers.Dense(300, activation = 'swish', name="layer21"),
            # tf.keras.layers.Dropout(0.25),
            # tf.keras.layers.Dense(300, activation = 'sigmoid', name="layer22"),
            # tf.keras.layers.Dropout(0.25),
            # # tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Dense(300, activation = 'swish', name="layer23"),
            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(300, activation = 'sigmoid', name="layer25"),
            tf.keras.layers.Dense(5,activation='sigmoid', name="final"),
        ]
)


opt = tf.keras.optimizers.SGD(
    learning_rate=0.0001, momentum=0.0, nesterov=False, name="SGD")

model.summary()

# model.compile('adam', loss='mse')
model.compile('adam', loss='binary_crossentropy')
#return model.summary()
history = model.fit(x_train,x_test,epochs = 20,validation_data =(y_train,y_test))
plt.plot(history.history['loss'])
# sc,acc = model.evaluate(InputN,LabelN)
# print(sc)
# print(acc)
#ans = model.predict(Input[777:776,:])

x = NPRSN.size


result = np.empty([x,5], dtype=float)
resultN = np.empty(x, dtype=float)
result = model.predict(InputF)
    
    
for iii in range(0,x):
    resultN[iii] = np.argmax(result[iii,:])
    
plt.plot(resultN)  
plt.plot(NPRSN)


# plt.plot(result[0:x,0])
# plt.plot(result[0:x,1])
# plt.plot(result[0:x,2])
# plt.plot(result[0:x,3])
# plt.plot(result[0:x,4])
# b = np.empty([1,104,1], dtype=float)
# a = np.empty([101])

# c= InputN[800:801,:]
# b[1,:,1] = c
# b = b.reshape(-1,104,1)
# a[:] = model.predict(c)


# result_line = np.empty(2000)
# for iii in range(0,2000):
#     result_line[iii] = np.where(result[iii,:] == np.amax(result[iii,:]))

