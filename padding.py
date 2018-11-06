# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:52:34 2018

@author: lenovo
"""
import matplotlib.pyplot as plt
from scipy import misc
from sklearn import preprocessing
from PIL import Image as im
import numpy as np
import os
import imageio
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
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

    pad_row = int((192 - input_row)*0.5)
    pad_col =int((192 - input_col)*0.5)
    #int(pad_col)
    print("pad_row: ", pad_row)
    print("pad_col", pad_col)
    
    pad_images = np.pad(image_input, ((pad_row, pad_row), (pad_col, pad_col)),  mode = "constant")
    #res_images = np.resize(pad_images, (pad_images.shape[0], 182))
    print("\\n---------------", pad_images.shape)
    return pad_images

def padding1(image_input):
    input_row = image_input.shape[0]
    input_col = image_input.shape[1]
    #image_input = np.resize(image_input, (input_row, input_col))

    pad_row = input_row
    pad_col =int((192 - input_col))
    #int(pad_col)
    print("pad_row: ", pad_row)
    print("pad_col", pad_col)
    if pad_col == 1:
        
        image_input = np.append(arr = np.zeros((pad_row, 1)).astype(int), values = image_input, axis = 1)
    #res_images = np.resize(pad_images, (pad_images.shape[0], 182))
    print("\\n---------------", image_input.shape)
    return image_input


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


path = "./radon_images1/"
createFolder('./paded_images1/')
pad_image1 = "./paded_images1/"
images = os.listdir(path)
for file in images:
    #nput_image = im.open(path+file)
    #input_image.load()
    input_image = misc.imread(path+file)
    gray_image = rgb2gray(input_image)  
    print(gray_image.shape)
    pad_images =  padding(gray_image)
    pad_images = padding1(pad_images)
    # pad_images.imsave(pad_image+file)
    print("\n--------------")
    print("\n-------paded images_shape::::::::::::----------", pad_images.shape)
    plt.imshow(pad_images)
    imageio.imwrite(pad_image1+file,pad_images)

i_path = './inverse_radon_images/'
createFolder('./i_paded_images/')
i_paded_image = './i_paded_images/'
i_images = os.listdir(i_path)
for i in i_images:
    i_image = misc.imread(i_path+i)
    i_gray = rgb2gray(i_image)
    i_pad = padding(i_gray)
    imageio.imwrite(i_paded_image+i, i_pad)

images_list = []
train_path = "./paded_images1/"
listing = os.listdir(train_path)
print(len(listing))
for file in listing:
    train_images = misc.imread(train_path+file)
    train_image = np.asarray(train_images).astype('float32')
    images_list.append(train_image)
image_data = np.array(images_list)
image_data = image_data.astype('float32')
print("image_data _shape---", image_data.shape)
image_data = np.reshape(image_data, [image_data.shape[0], image_data.shape[1], image_data.shape[2], 1])
print("\nimage_data _shapereshaped-------", image_data.shape)


images_list1 = []
train_path1 = "./i_paded_images/"
listing = os.listdir(train_path1)
print(len(listing))
for file1 in listing:
    train_images1 = misc.imread(train_path1+file1)
    train_image1 = np.asarray(train_images1).astype('float32')
    images_list1.append(train_image1)
image_data = np.array(images_list1)
image_data1 = image_data.astype('float32')
print("image_data _shape---", image_data1.shape)
image_data1 = np.reshape(image_data1, [image_data1.shape[0], image_data1.shape[1], image_data1.shape[2], 1])
print("\nimage_data _shapereshaped-------", image_data1.shape)


createFolder('./datasets/trainy_data/')
padoutput_images = './padoutput_image/'
img1 = misc.imread('phantom128Head.png', flatten=True).astype('float64')
print("\n\\\\\\\\\\\\\\\\\\\\\\\\")
print(img1.shape)
pad_images2 =  padding(img1)
plt.imshow(pad_images2)
plt.figure()
createFolder('./padoutput_image/')
plt.imsave('./padoutput_image/pad_output_image_image', pad_images2)

"""
k=0
trainy_data = './datasets/trainy_data/'
trainy_data1 = []
listing = os.listdir(padoutput_images)
for k in range(0, 180):
    for file in listing:
        trainy_images1 = misc.imread(trainy_data+file)
        trainy_images1 = rgb2gray(trainy_images1)
        trainy_images1.imsave(trainy_data+file)
        trainy_images1 = np.asarray(trainy_images1).astype('float32')
        trainy_data1.append(trainy_images1)

trainy_data1 = np.array(trainy_data)
        
   """     