# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 09:49:02 2018

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:03:55 2018

@author: lenovo
"""

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
from keras.layers import Dropout, Conv2D, MaxPooling2D, BatchNormalization
from scipy import misc
from keras import layers
from keras.utils import plot_model
#import pydot_ng as pydot
#pydot.find_graphviz()
import numpy as np
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import keras.backend as K
K.set_image_data_format('channels_last')

#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


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



np.random.seed(0)






def unet(channel, height, width):
    inputs = Input((height, width, channel))
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(conv1)
    
    
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    

    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(pool3)
    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

   

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,2))(drop5))
    merge7 = concatenate([drop3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(conv7)

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer ='glorot_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(conv8)

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', data_format="channels_last", kernel_initializer = 'glorot_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'relu')(conv9)

    model = Model(input = inputs, output = conv10)

    

    return model


input_180 = np.load('input111_180.npy')
output_180 = np.load('output1_180.npy')


X_train, X_test, Y_train, Y_test = train_test_split(input_180, output_180, test_size=0.2)
model = unet(input_180.shape[3], input_180.shape[1], input_180.shape[2])
adam = optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon= None, decay=0.0, amsgrad=False)
model.compile(optimizer = adam, loss = 'mean_squared_error', metrics = ['accuracy'])
print(model.summary())
plot_model(model, to_file = 'shirisha.png')

model.fit(X_train, Y_train, epochs=50 , batch_size = 1, verbose = 2)
#model.fit(trainx_input_images, trainy_input_images, epochs = 5, batch_size = 1)
# plot metrics
#plt.plot(history.history['mean_squared_error'])

#plt.show()

score = model.evaluate(X_test, Y_test, verbose=2)
print('Test Loss :', score[0])
print('Test Accuracy: ', score[1]*100)
#test = X_test[1:2]
    #print(test_image.shape)
test = misc.imread('i145.png', flatten=True).astype('float64')
test = np.reshape(test, [1, test.shape[0], test.shape[1], 1])
result = model.predict(test)
reshape = np.reshape(result, [result.shape[1], result.shape[2]])
plt.imshow(reshape, cmap='gray')


"""
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
 

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

model.save("model.hdf5")
loaded_model = load_model("model.hdf5")

loaded_model.compile(optimizer = adam, loss = 'mean_squared_error', metrics = ['accuracy'])
print(loaded_model.summary())
loaded_model.fit(X_train, Y_train, epochs=10, batch_size = 2)

score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('Test Loss :', score[0])
print('Test Accuracy: ', score[1])
test = X_test[0:1]
   
result = loaded_model.predict(test)
reshape = np.reshape(result, [result.shape[1], result.shape[2]])

plt.imshow(reshape, cmap='gray')


#model.fit(trainx_input_images, trainy_input_images, epochs = 5, batch_size = 1)




#plt.imshow(new_image[0,:,:,0],cmap='gray');
resulted_image = np.resize(new_image, [new_image.shape[1], new_image.shape[2]])
#plt.imshow(new_image, cmap = "gray")


"""