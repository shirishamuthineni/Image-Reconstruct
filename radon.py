import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc
from skimage.transform import radon, iradon

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. '+ directory)
    

    
image = misc.imread('phantom128Head.png', flatten=True).astype('float64')
print (image.shape)

list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in range(0, 10):
    theta = np.linspace(0, 180, 180/list[i])
    
    createFolder('./radon_images/')
    sinogram = radon(image, theta=theta)
    plt.xlabel("Numbers of Projections");
    plt.ylabel("Projection position (pixel)");
    plt.imshow(sinogram, cmap='gray')
    fig = plt.figure()
    plt.imsave('./radon_images/sinogram%d_image'%i, sinogram)
    plt.xlabel("Number of Projections");
    plt.ylabel("Projection position(pixel)");
    createFolder('./inverse_radon_images/')
    ireconstruct = iradon(sinogram, theta=theta)
    plt.imshow(ireconstruct, cmap='gray')
    plt.figure()
    plt.imsave('./inverse_radon_images/iradon%d_image'%i, ireconstruct)
    
    
path1 = "./radon_images/"
path2 = './flip_image/'
listing = os.listdir(path1)
for file in listing:
    createFolder('./flip_image/')
    image = Image.open(path1+file)
    left_right = image.transpose(Image.FLIP_LEFT_RIGHT)
    left_right.save(path2+file)
    
    
path3 = './flip_top_down/'

for file in listing:
    createFolder('./flip_top_down/')
    image = Image.open(path1+file)
    top_down = image.transpose(Image.FLIP_TOP_BOTTOM)
    top_down.save(path3+file)
  
    
path4 = './rotate_45_degrees/'
degrees_to_rotate = 45;
for file in listing:
    createFolder('./rotate_45_degrees/')
    image = Image.open(path1+file)
    rotate = image.rotate(degrees_to_rotate)
    rotate.save(path4+file)
    
    
path5 = './rotate_90_degrees/'
degrees_to_rotate = 90;
for file in listing:
    createFolder('./rotate_90_degrees/')
    image = Image.open(path1+file)
    rotate = image.rotate(degrees_to_rotate)
    rotate.save(path5+file)
    
"""path6 = './noise_images/'
for file in listing:
    createFolder('./noise_images/')
    image = Image.open(path1+file)
    noise_image = image + 3 * image.std() * np.random.random(image.shape)
    noise_image.show()
    noise_image.save(path6+file)
"""
