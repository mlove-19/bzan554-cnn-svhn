from tokenize import PlainToken
import numpy as np
import os
#import PIL
#from PIL import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

# os.chdir("/home/users/mlove/MSBA/BZAN 554 Deep Learning/Projects/Project 3")

#create training data list
os.chdir('C:/Users/jamfo/Documents/Deep Learning/bzan554-cnn-svhn/train')
len([name for name in os.listdir('.') if os.path.isfile(name)]) #how many files in folder
os.getcwd()

ims_train=[]

for i in range(1,1000):
    img=mpimg.imread(str(i)+'.png')
    ims_train.append(img)

plt.imshow(ims_train[10])
plt.show()


#create testing data list
os.chdir('C:/Users/jamfo/Documents/Deep Learning/bzan554-cnn-svhn/test')
len([name for name in os.listdir('.') if os.path.isfile(name)]) #how many files in folder
os.getcwd()

ims_test=[]

for i in range(1,1000):
    img=mpimg.imread(str(i)+'.png')
    ims_test.append(img)

plt.imshow(ims_test[2])
plt.show()


######################
### Project Phases ###
######################

# Read in data
# Padding
# Rescaling RGB values
# Code examples: image_dense.py, images_convnet.py (from Dropbox)