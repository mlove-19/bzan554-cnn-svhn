from tokenize import PlainToken
import numpy as np
import os
#import PIL
#from PIL import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import json
from os import listdir
from os.path import isfile, join
import pandas as pd
# os.chdir("/home/users/mlove/MSBA/BZAN 554 Deep Learning/Projects/Project 3")


### Add your folder paths here##

test_path_jake = 'C:/Users/jamfo/Documents/Deep Learning/bzan554-cnn-svhn/test'
train_path_jake = 'C:/Users/jamfo/Documents/Deep Learning/bzan554-cnn-svhn/train'


test_path_cole = 'C:/Users/cole\Documents/Spring MSBA/Deep Learning/SVHN/test'
train_path_cole='C:/Users/cole\Documents/Spring MSBA/Deep Learning/SVHN/train'


test_path_marisa = '/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/test'
train_path_marisa = '/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/train'

### Set your paths here####
train_path = train_path_marisa
test_path = test_path_marisa


# set directory to train
os.chdir(train_path)
len([name for name in os.listdir('.') if os.path.isfile(name)]) #how many files in folder
os.getcwd()

# get list of files in train folder since their numbers are not sequential
train_file_names = [f for f in listdir(train_path) if isfile(join(train_path, f))]

ims_train=[]

for i in range(len(train_file_names)):
    img=mpimg.imread(train_file_names[i] + '.png')
    ims_train.append(img)

train_file_names[0]
plt.imshow(ims_train[0])
plt.show()



#create testing data list
os.chdir(test_path)

len([name for name in os.listdir('.') if os.path.isfile(name)]) #how many files in folder
os.getcwd()

ims_test=[]

for i in range(1,1000):
    img=mpimg.imread(str(i)+'.png')
    ims_test.append(img)

plt.imshow(ims_test[1])
plt.show()


### read in json TEST digit structure to df 
with open(test_path+'/digitStruct.json', 'r') as f:
  test_digits = json.load(f)

test_df = pd.json_normalize(test_digits, 'boxes', 'filename', 
                    record_prefix='boxes_')

test_df.locations = pd.DataFrame(test_df.locations.values.tolist())['name']
test_df = test_df.groupby('filename')['boxes'].apply(','.join).reset_index()
print (test_df)



### read in json TRAIN digit structure to df 
with open(train_path+'/digitStruct.json', 'r') as f:
  train_digits = json.load(f)

train_df = pd.json_normalize(train_digits, 'boxes', 'filename', 
                    record_prefix='boxes_')

train_df.locations = pd.DataFrame(train_df.locations.values.tolist())['name']
train_df = train_df.groupby('filename')['boxes'].apply(','.join).reset_index()
print (train_df)



######################
### Project Phases ###
######################

# Read in data
# Padding
# Rescaling RGB values
# Code examples: image_dense.py, images_convnet.py (from Dropbox)