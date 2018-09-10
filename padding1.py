# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:52:34 2018

@author: lenovo
"""
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import numpy as np
import os
import imageio
import matplotlib.image as mpimg

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. '+ directory)
def padding(image_input):
    input_row = image_input.shape[0]
    input_col = image_input.shape[1]
    #image_input = np.resize(image_input, (input_row, input_col))

    pad_row = (182 - input_row)
    pad_col =int((182 - input_col)*0.5)
    #int(pad_col)
    print("pad_row: ", pad_row)
    print("pad_col", pad_col)
    
    pad_images = np.pad(image_input, ((pad_row, pad_row), (pad_col, pad_col)),  mode = "constant")
    #res_images = np.resize(pad_images, (pad_images.shape[0], 182))
    print("\\n---------------", pad_images.shape)
    return pad_images

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

path = "./radon_images/"
createFolder('./paded_images/')
pad_image = "./paded_images/"
images = os.listdir(path)
for file in images:
    # input_image = Image.open(path+file)
    input_image = misc.imread(path+file)
    gray_image = rgb2gray(input_image)  
    print(gray_image.shape)
    #plt.imshow(gray_image)
    pad_images =  padding(gray_image)
    # pad_images.imsave(pad_image+file)
    print("\n--------------")
    print("\n-----------------", pad_images.shape)
    plt.imshow(pad_images)
    imageio.imwrite(pad_image+file,pad_images)
