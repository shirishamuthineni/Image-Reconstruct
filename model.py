# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:18:40 2018

@author: lenovo
"""


import matplotlib.pyplot as plt
import os
#from PIL import Image
#import keras.utils 
from keras.layers import Input, UpSampling2D
from keras.layers import concatenate, initializers
from keras.models import Sequential, Model
from keras.layers import Dropout, Conv2D, MaxPooling2D
from scipy import misc
from keras import layers
from keras.utils import plot_model
import numpy as np
import keras.backend as K
K.set_image_data_format('channels_last')


#import numpy as np
#from keras import optimisers
from keras.optimizers import Adam
import tensorflow as tf
#import keras
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. '+ directory)

res = []
size = 0
np.random.seed(0)       
image = misc.imread('sinogram0_image.png', flatten=True).astype('float64')
test_image = misc.imread('./radon_images/sinogram1_image.png', flatten=True).astype('float64')
def padding1(initial_image):
    in_rows, in_col = initial_image.shape
    pad_row = 182 - in_rows
    pad_col = 182 - in_col
    res_row = in_rows+pad_row
    res_col = in_col+pad_col
    res = np.zeros((res_row, res_col))
    res[size: res_row - pad_row, size: res_col - pad_col] = initial_image
    return res
test_image = padding1(test_image)
test_image = np.resize(test_image, [1, test_image.shape[0], test_image.shape[1], 1])
print("\n---Test image.shape ----", test_image.shape)
print("\n Test image shape----------")
print("\n-----test image rows :---", test_image.shape[0])
print("\n-----test image columns :---", test_image.shape[1])
#print("\n-----test image channels :---", test_image.shape[3])
#print (image.shape)
def padding(initial_array, pad_size):
    in_row, in_col = initial_array.shape
    res_row, res_col = in_row, in_col+pad_size*2
    
    #res[1000][1000]
    res = np.zeros((res_row, res_col))
    
    res[size: res_row, size: res_col-2] = initial_array

    #print(res)
    return res
def paddingO(initial_array, pad_size):
    in_row, in_col = initial_array.shape
    res_row, res_col = in_row+pad_size*2, in_col+pad_size*2
    
    #res[1000][1000]
    res = np.zeros((res_row, res_col))
    
    res[size: res_row-2*pad_size, size: res_col-2*pad_size] = initial_array

    print(res)
    return res
    
image = padding(image, 1)
train_x = np.resize(image, (1, image.shape[0], image.shape[1], 1))
print("\n--------Train Image Rows ---", train_x.shape[1])
print("\n--------train_x col------", train_x.shape[2])
print("\n--------train_x no of images------", train_x.shape[0])
print("\n--------train_x cchannels------", train_x.shape[3])
"""img1 = np.load('sinogram_image_array.npy');
plt.imshow(img1)
img = tf.reshape(img1, [1, img1.shape[0], img1.shape[1], 1])
np.save('reshape_x', img);
img2 = np.load('reshape_x.npy')
rows = img2.shape[1]
col = img2.shape[2]
print("/n-----rws:-------", rows)
print("\n--------columns-------", col)
#no_of _channel = re_img.shape[3]
print("sinogram _ reshape ", img.shape)

"""



#print(img.shape)
#path = "./datasets/train_data/"
#x_train = os.listdir(path)
y_train = misc.imread('phantom128Head.png', flatten=True).astype('float64')
y_train = paddingO(y_train, 24)
row, cols = y_train.shape
train_y = np.resize(y_train, [1, row, cols, 1])
#model = Sequential()

def unet(channel, height, width):
    inputs = Input((height, width, channel))
    print("--------inputs------")
    print(inputs)
    conv1 = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', data_format="channels_last", activation = 'relu', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', data_format="channels_last", activation = 'relu', kernel_initializer = 'he_normal')(conv1)
    conv11 = MaxPooling2D(pool_size = (2, 2), strides = (1, 1))(conv1)
    conv11 = MaxPooling2D(pool_size = (2, 2), strides = (1, 1))(conv11)
    for i in range(1, 3):
        conv11 = MaxPooling2D(pool_size = (2, 2), strides = (1, 1))(conv11)
        conv11 = MaxPooling2D(pool_size = (2, 2), strides = (1, 1))(conv11)
    print("----\nConvolution 1 dimensions''''''''''")
    print(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("\n-----------------pooling 1 dimension--------------")
    pool1 = MaxPooling2D(pool_size = (2, 2), strides = (1, 1))(pool1)
    print(pool1)


    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(conv2)
    print("\n-----------convolution 2 ")
    print(conv2)
    conv21 = MaxPooling2D(pool_size = (2, 2), strides = (1, 1))(conv2)
    conv21 = MaxPooling2D(pool_size = (2, 2), strides = (1, 1))(conv21)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("\n-------------pooling 2----------")
    pool2 = MaxPooling2D(pool_size = (2, 2), strides = (1, 1))(pool2)
    print(pool2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(conv3)
    print("\n-----------convolution 3 ")
    print(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print("\n-------------pooling 3----------")
    print(pool3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(conv4)
    print("\n-----------convolution 4 ")
    print(conv4)
    drop4 = Dropout(0.5)(conv4)
    print("\n----------------drop4:------")
    print(drop4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    print("\n-------------pooling 4----------")
    print(pool4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(conv5)
    #pool5 = MaxPooling2D(pool_size=(2, 2), strides = (1, 1))(conv5)
    #pool6 = MaxPooling2D(pool_size=(2, 2), strides = (1, 1))(pool5)
    #pool7 = MaxPooling2D(pool_size=(2, 2), strides = (1, 1))(pool6)
    print("\n-----------convolution 5 ")
    print(conv5)
    drop5 = Dropout(0.5)(conv5)
    print(UpSampling2D(size = (2, 2))(drop5))
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    print("\n-------up6:--------")
    print(up6)
    merge6 = concatenate([drop4,up6], axis = 3)
    print("\n------------merge 6:-------")
    print(merge6)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(conv6)
    print("\n-----------convolution 6------ ")
    print(conv6)
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    print("\n---------up7-------")
    print(up7)
    merge7 = concatenate([conv3,up7], axis=3)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(conv7)
    print("\n-----------convolution 7---------- ")
    print(conv7)
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    
    merge8 = concatenate([conv21, up8], axis = 3)


    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(conv8)
    print("\n-----------convolution 8---------- ")
    print(conv8)
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    
    merge9 = concatenate([conv11, up9], axis = 3)
   


    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'he_normal')(conv9)
    print("\n-----------convolution 9-------- ")
    print(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    print("\n-----------convolution 10--------")
    print(conv10)
    model = Model(input = inputs, output = conv10)

   
    return model

model = unet(train_x.shape[3], train_x.shape[1], train_x.shape[2])
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(train_x, train_y, epochs = 100, batch_size = 1)
new_image = model.predict(test_image)

plt.imshow(new_image[0,:,:,0],cmap='gray');
new_image = np.resize(new_image, [new_image.shape[1], new_image.shape[2]])





"""
path = "./datasets/train_data/"
path1 = "./datasets/test_data/"
listing = os.listdir(path)
for file in listing:
    image = Image.open(path+file)
    rows, col = image.shape
    re_image = tf.reshape(image, [1, rows, col, 1])
    n, r_rows, r_col, channels = re_image.shape
    print(r_rows)
    print(n)

    print(channels)
    print(r_col)
   """  
