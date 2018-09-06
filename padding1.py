# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:52:34 2018

@author: lenovo
"""
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np  
image = misc.imread('sinogram0_image.png', flatten=True).astype('float64')
print (image.shape)

res = []
size = 0


def paddingO(initial_array, pad_size):
    in_row, in_col = initial_array.shape
    print("\----In_columns--------")
    print(in_col)
    res_row, res_col = in_row+pad_size*2, in_col+pad_size*2
    
    #res[1000][1000]
    res = np.zeros((res_row, res_col))
    
    res[size: res_row-2*pad_size, size: res_col-2*pad_size] = initial_array

    
    return res
y_train = misc.imread('phantom128Head.png', flatten=True).astype('float64')
print("\n--------------------Before padding:---------------")
plt.imshow(y_train, cmap="gray")
y_train = paddingO(y_train, 24)
row, cols = y_train.shape
print(row)
print(cols)
print("\n----------------After Padding-------------")
plt.imshow(y_train, cmap="gray")