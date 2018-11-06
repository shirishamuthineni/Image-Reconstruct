# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 23:20:50 2018

@author: lenovo
"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image as im
import glob
import os




FILES = glob.glob(r'C:\Users\lenovo\Desktop\image_example\datasets\trainy_data\*.png')
#listingy = os.listdir(trainy_image)
trainy_images = []
for i in range(1, 181):
    for file in FILES:
        imagey = im.open(file)
        imagey.load()
        train_datay = np.asarray(imagey, dtype='float32')
        trainy_images.append(train_datay)    

trainy_images = np.array(trainy_images)
trainy_reshape_images = np.resize(trainy_images, [trainy_images.shape[0], trainy_images.shape[1], trainy_images.shape[2], 1])
np.load('train_y_192x192PADDEDFBPphantom.npy')