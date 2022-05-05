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
import cv2
# os.chdir("/home/users/mlove/MSBA/BZAN 554 Deep Learning/Projects/Project 3")


### Add your folder paths here##

test_path_jake = 'C:/Users/jamfo/Documents/Deep Learning/SVHN/test'
train_path_jake = 'C:/Users/jamfo/Documents/Deep Learning/SVHN/train'
main_path_jake= 'C:/Users/jamfo/Documents/Deep Learning/SVHN'
train_crop_jake= 'C:/Users/jamfo/Documents/Deep Learning/SVHN/train_cropped_images'
train_padded_jake= 'C:/Users/jamfo/Documents/Deep Learning/SVHN/train_images_padded'
test_crop_jake= 'C:/Users/jamfo/Documents/Deep Learning/SVHN/test_cropped_images'
test_padded_jake= 'C:/Users/jamfo/Documents/Deep Learning/SVHN/test_images_padded'
train_grey_jake= 'C:/Users/jamfo/Documents/Deep Learning/SVHN/train_images_grey'
test_grey_jake= 'C:/Users/jamfo/Documents/Deep Learning/SVHN/test_images_grey'


test_path_cole = 'C:/Users/cole\Documents/Spring MSBA/Deep Learning/SVHN/test'
train_path_cole='C:/Users/cole\Documents/Spring MSBA/Deep Learning/SVHN/train'


test_path_marisa = '/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/test'
train_path_marisa = '/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/train'

### Set your paths here####
train_path = train_path_jake
test_path = test_path_jake



###################### TRAIN ####################

### read in json TRAIN digit structure to df 
with open(train_path+'/digitStruct.json', 'r') as f:
  train_digits = json.load(f)

train_df = pd.json_normalize(train_digits, 'boxes', 'filename', 
                    record_prefix='boxes_')

print (train_df)

train_df.rename(columns = {'boxes_top':'y_top','boxes_left':'x_left','boxes_width':'width','boxes_height':'height'}, inplace = True)



#Get toal width of image
width_df = train_df[['width','filename']]
total_width = width_df.groupby('filename',sort=False).sum().reset_index()

#Get total height of image
height_df = train_df[['height','filename']]
total_height = height_df.groupby('filename',sort=False).max().reset_index()

#Get top-left Y coordinate for image
y_max_df = train_df[["y_top","filename"]]
y_max_coor_df = y_max_df.groupby('filename', sort=False).min().reset_index() #min because of how the image is plotted
y_max_coor_df.rename(columns={'y_top':'y_max'})

#Get top-left X coordinate for image
x_df = train_df[["x_left","filename"]]
x_max_coor_df = x_df.groupby('filename', sort = False).min().reset_index()  #min because of how the image is plotted
x_max_coor_df.rename(columns={'x_left':'x_max'})

#Merge into one
new_train_df = total_width
new_train_df = new_train_df.merge(total_height,on='filename')
new_train_df = new_train_df.merge(y_max_coor_df,on='filename')
new_train_df = new_train_df.merge(x_max_coor_df,on='filename')

#Find bottom-right Y coordinate for image
new_train_df['y_bottom'] = (new_train_df['y_top'] + new_train_df['height'])


#Find bottom-right X coordinate for image
new_train_df['x_bottom'] = (new_train_df['x_left'] + new_train_df['width'])


#creating joined labels

new_train_df["label"] = 0
new=pd.DataFrame(train_df.groupby('filename',sort=False).agg(list)).reset_index()
new=new[['filename','boxes_label']]
new_train_data = new_train_df.merge(new,on='filename')

for i in range(len(new_train_data)):
#for i in range(25):
  filename=new['filename'][i]
  num=new_train_data['boxes_label'][i]
  num = [ int(x) for x in num ]
  for j in range(len(num)):
    if num[j]==10:
      num[j]=0
    else:
      num[j]=num[j]
  num = int(''.join(map(str,num)))
  new_train_data.loc[ new_train_data['filename'] == filename, 'label']=num 

df_train = new_train_data[['filename','label']]



########################### TEST ####################
### read in json TEST digit structure to df 
with open(test_path+'/digitStruct.json', 'r') as f:
  test_digits = json.load(f)

test_df = pd.json_normalize(test_digits, 'boxes', 'filename', 
                    record_prefix='boxes_')


print (test_df)

test_df.rename(columns = {'boxes_top':'y_top','boxes_left':'x_left','boxes_width':'width','boxes_height':'height'}, inplace = True)



#Get toal width of image
width_df = test_df[['width','filename']]
total_width = width_df.groupby('filename',sort=False).sum().reset_index()

#Get total height of image
height_df = test_df[['height','filename']]
total_height = height_df.groupby('filename',sort=False).max().reset_index()

#Get top-left Y coordinate for image
y_max_df = test_df[["y_top","filename"]]
y_max_coor_df = y_max_df.groupby('filename', sort=False).min().reset_index() #min because of how the image is plotted
y_max_coor_df.rename(columns={'y_top':'y_max'})

#Get top-left X coordinate for image
x_df = test_df[["x_left","filename"]]
x_max_coor_df = x_df.groupby('filename', sort = False).min().reset_index()  #min because of how the image is plotted
x_max_coor_df.rename(columns={'x_left':'x_max'})

#Merge into one
new_test_df = total_width
new_test_df = new_test_df.merge(total_height,on='filename')
new_test_df = new_test_df.merge(y_max_coor_df,on='filename')
new_test_df = new_test_df.merge(x_max_coor_df,on='filename')

#Find bottom-right Y coordinate for image
new_test_df['y_bottom'] = (new_test_df['y_top'] + new_test_df['height'])


#Find bottom-right X coordinate for image
new_test_df['x_bottom'] = (new_test_df['x_left'] + new_test_df['width'])


#creating joined labels

new_test_df["label"] = 0
new=pd.DataFrame(test_df.groupby('filename',sort=False).agg(list)).reset_index()
new=new[['filename','boxes_label']]
new_test_data = new_test_df.merge(new,on='filename')

for i in range(len(new_test_data)):
#for i in range(25):
  filename=new['filename'][i]
  num=new_test_data['boxes_label'][i]
  num = [ int(x) for x in num ]
  for j in range(len(num)):
    if num[j]==10:
      num[j]=0
    else:
      num[j]=num[j]
  num = int(''.join(map(str,num)))
  new_test_data.loc[ new_test_data['filename'] == filename, 'label']=num 

df_test = new_test_data[['filename','label']]

path_repo = 'C:/Users/jamfo/Documents/Deep Learning/bzan554-cnn-svhn'
os.chdir(path_repo)
os.getcwd()

df_train.to_csv('train_labels.csv',index=False)
df_test.to_csv('test_labels.csv',index=False)




