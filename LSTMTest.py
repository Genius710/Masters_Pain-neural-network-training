# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:53:05 2020

@author: Povilas-Predator-PC
"""

from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np

#data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
# arr = [1, 2, 3, 4, 5, 6]
form = [5, 6, 7, 8, 9, 10, 12, 16, 17, 20]
iii =1

num = str(form[iii])

mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'CNAPBlock.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
InputN = mat_contents['Input']
LabelN = mat_contents['Label']
NPRSN = mat_contents['NPRS']

# for iii in range(2,6):
#     num = str(form[iii])
#     print('Aurimod0' +num +'CNAPBlock.mat')
#     mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'CNAPBlock.mat')
#     mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
#     Input = mat_contents['Input']
#     Label = mat_contents['Label']
#     NPRS = mat_contents['NPRS']
#     InputN = np.vstack((InputN,Input))
#     LabelN = np.vstack((LabelN,Label))
#     NPRSN = np.hstack((NPRSN,NPRS))

[x,y] = NPRSN.shape
NPRSNF = np.empty([y,5], dtype=float)
NPRSN =NPRSN[0]
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
    







iii =7
num = str(form[iii])
mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'CNAPBlock.mat')
mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
InputV = mat_contents['Input']
LabelV = mat_contents['Label']
NPRSV = mat_contents['NPRS']

# for iii in range(8,9):
#     num = str(form[iii])
#     print('Aurimod0' +num +'CNAPBlock.mat')
#     mat_fname = pjoin('D:/OneDrive - Kaunas University of Technology/MAGISTRINIS/Data/', 'Aurimod0' +num +'CNAPBlock.mat')
#     mat_contents = sio.loadmat(mat_fname,struct_as_record=0)
#     Input = mat_contents['Input']
#     Label = mat_contents['Label']
#     NPRS = mat_contents['NPRS']
#     InputV = np.vstack((InputV,Input))
#     LabelV = np.vstack((LabelV,Label))
#     NPRSV = np.hstack((NPRSV,NPRS))

[x,y] = NPRSV.shape
NPRSVF = np.empty([y,5], dtype=float)
NPRSV = NPRSV[0]
for iii in range(0,y):
    if NPRSV[iii] ==0:
        NPRSVF[iii,0] = 1
        NPRSVF[iii,1] = 0
        NPRSVF[iii,2] = 0
        NPRSVF[iii,3] = 0
        NPRSVF[iii,4] = 0
    elif NPRSV[iii] > 0 and NPRSV[iii] <25: 
        NPRSVF[iii,0] = 0
        NPRSVF[iii,1] = 1
        NPRSVF[iii,2] = 0
        NPRSVF[iii,3] = 0
        NPRSVF[iii,4] = 0
    elif NPRSV[iii] >= 25 and NPRSV[iii] <50:
        NPRSVF[iii,0] = 0
        NPRSVF[iii,1] = 0
        NPRSVF[iii,2] = 1
        NPRSVF[iii,3] = 0
        NPRSVF[iii,4] = 0
    elif NPRSV[iii] >= 50 and NPRSV[iii] <75:
        NPRSVF[iii,0] = 0
        NPRSVF[iii,1] = 0
        NPRSVF[iii,2] = 0
        NPRSVF[iii,3] = 1
        NPRSVF[iii,4] = 0
    elif NPRSV[iii] >= 75:
        NPRSVF[iii,0] = 0
        NPRSVF[iii,1] = 0
        NPRSVF[iii,2] = 0
        NPRSVF[iii,3] = 0
        NPRSVF[iii,4] = 1

InputV = InputV.reshape(-1,104,1)
LabelV = LabelV.reshape(-1,101,1)

InputN = InputN.reshape(-1,104,1)
LabelN = LabelN.reshape(-1,101,1)

NPRSNF = NPRSNF.reshape(-1,5,1)
NPRSVF = NPRSVF.reshape(-1,5,1)

import tensorflow as tf
import keras



#def mod(y_new):
    
model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(104, input_shape=[104,1], name="layer1"),
            # tf.keras.layers.LSTM(100),
            tf.keras.layers.LSTM(500),
            # tf.keras.layers.LSTM(100),

            tf.keras.layers.Dense(101,activation='softmax', name="layer3"),
        ]

)
# opt = tf.keras.optimizers.SGD(
#     learning_rate=0.1, momentum=0.0, nesterov=False, name="SGD")

model.summary()

model.compile('adam', loss='mean_squared_error')
#return model.summary()
model.fit(InputN,LabelN,epochs = 5,validation_data =(InputV,LabelV))

sc,acc = model.evaluate(InputN,LabelN)
print(sc)
print(acc)
#ans = model.predict(Input[777:776,:])

[x] = NPRSN.shape

result = np.empty([x,101], dtype=float)
# for iii in range(600,800):
#     print(iii)
#     print(InputN[0+iii:1+iii,:])
#     b = InputN[0+iii:1+iii,:]
#     result[iii,:] = model.predict(b)


b = np.empty([1,104,1], dtype=float)
a = np.empty([101])

c= InputN[800:801,:]
# b[1,:,1] = c
# b = b.reshape(-1,104,1)
a[:] = model.predict(c)


# result_line = np.empty(2000)
# for iii in range(0,2000):
#     result_line[iii] = np.where(result[iii,:] == np.amax(result[iii,:]))

# import matplotlib.pyplot as plt
# plt.plot(result_line)


#prediction = house_model([7.0])
#print(prediction)