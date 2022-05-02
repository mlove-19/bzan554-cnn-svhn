from enum import unique
from fileinput import filename
from multiprocessing.sharedctypes import Array
from operator import concat
from tokenize import PlainToken
from unicodedata import digit
import numpy as np
import os
#import PIL
#from PIL import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import json
from os import listdir, rename
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
train_path = train_path_jake
test_path = test_path_jake


# set directory to train
os.chdir(train_path)
len([name for name in os.listdir('.') if os.path.isfile(name)]) #how many files in folder
os.getcwd()

# get list of files in train folder since their numbers are not sequential
train_file_names = [f for f in listdir(train_path) if isfile(join(train_path, f))]

ims_train=[]

for i in range(len(train_file_names)):
    img=mpimg.imread(train_file_names[i])
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
    img=mpimg.imread(str(i))
    ims_test.append(img)

plt.imshow(ims_test[1])
plt.show()


### read in json TEST digit structure to df 
with open(test_path+'/digitStruct.json', 'r') as f:
  test_digits = json.load(f)

test_df = pd.json_normalize(test_digits, 'boxes', 'filename', 
                    record_prefix='boxes_')


print (test_df)



### read in json TRAIN digit structure to df 
with open(train_path+'/digitStruct.json', 'r') as f:
  train_digits = json.load(f)

train_df = pd.json_normalize(train_digits, 'boxes', 'filename', 
                    record_prefix='boxes_')

print (train_df)


train_df_new = pd.DataFrame(train_df['filename'].value_counts().rename_axis('filename').reset_index(name='digit_length'))


#Finding min and max coordinated for complete bounding box
y_df = train_df[["boxes_top","filename"]]
y_max_coor_df = y_df.groupby('filename').max().reset_index()
y_min_coor_df = y_df.groupby('filename').min().reset_index()



x_df = train_df[["boxes_left","filename"]]
x_max_coor_df = x_df.groupby('filename').max().reset_index()
x_min_coor_df = x_df.groupby('filename').min().reset_index()

#Joining data 
new_data = y_max_coor_df.merge(y_min_coor_df,on='filename')
new_data = new_data.merge(x_max_coor_df,on='filename')
new_data = new_data.merge(x_min_coor_df,on='filename')
new_data = new_data.merge(train_df_new,on='filename')


#creating joined labels
new=new_data
new["label"] = 0

for i in range(len(new_data)):
#for i in range(15):
  filename=new['filename'][i]
  subset=train_df[train_df['filename']==filename]
  subset=subset.sort_values('boxes_left',axis=0,ascending=True).reset_index()
  num=[]
  for i in range(len(subset)):
    if subset['boxes_label'][i]==10:
      num.append(0)
    else:
      num.append(subset['boxes_label'][i])
  num = [ int(x) for x in num ]
  num = int(''.join(map(str,num)))  
  new.loc[ new['filename'] == filename, 'label']=num 

new_data=new

######################
### Project Phases ###
######################

# Read in data
# Padding
# Rescaling RGB values
# Code examples: image_dense.py, images_convnet.py (from Dropbox)