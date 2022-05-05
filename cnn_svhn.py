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
    img=mpimg.imread(str(i)+'.png')
    ims_test.append(img)

plt.imshow(ims_test[0])
plt.show()

#create testing data list
os.chdir(train_path)

len([name for name in os.listdir('.') if os.path.isfile(name)]) #how many files in folder
os.getcwd()

ims_train=[]

for i in range(1,1000):
    img=mpimg.imread(str(i)+'.png')
    ims_train.append(img)

plt.imshow(ims_train[251])
plt.show()

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




os.chdir("/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3")
new_train_data.to_csv('train_bounding_box.csv',index=False)

# set working directory to folder with bounding box labels csv
os.getcwd()
os.chdir("/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3")

# make folder to store cropped images
os.mkdir("train_cropped_images")

# read in labels csv
df = pd.read_csv("train_bounding_box.csv", index_col = False)
print(df.head())

# set working directory to folder with check images
os.getcwd()
os.chdir('/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/train')

# setting path of output folder
path = "/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/train_cropped_images" 

#Taking out -1 left value and replacing it with 0 
df[['x_left']] = df[['x_left']].clip(lower = 0)


# cropping
for i in range(len(df)):
  img = cv2.imread(df.filename[i]) # read in image
      # crop image using coordinates from df
  cropped_image = img[int(df.y_top[i]):int(df.y_bottom[i]), int(df.x_left[i]):int(df.x_bottom[i])] # img[ymin:ymax, xmin:xmax]

      # save cropped image to output folder
  cv2.imwrite(os.path.join(path , df.filename[i]), cropped_image)


############################################################

# Padding Cropped Images
# Code adapted from https://stackoverflow.com/a/59698237

# make folder to store padded images
os.chdir("/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3")
os.mkdir("train_images_padded")

# change working directory to cropped images folder
os.chdir("/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/train_cropped_images")

# setting path of output folder
path =  "/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/train_images_padded"


for i in range(len(df)):
    img = cv2.imread(df.filename[i]) # read in cropped image - can use same df and filnename column since names are the same, just in a different folder
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color (white) for padding
    new_image_width = int(max(df.width))
    new_image_height = int(max(df.height))
    color = (255,255,255)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
        x_center:x_center+old_image_width] = img

    """ # view result
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows() """

    # save result
    cv2.imwrite(os.path.join(path , df.filename[i]), result)







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
new=pd.DataFrame(train_df.groupby('filename',sort=False).agg(list)).reset_index()
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




os.chdir("/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3")
new_test_data.to_csv('test_bounding_box.csv',index=False)

# set working directory to folder with bounding box labels csv
os.getcwd()
os.chdir("/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3")

# make folder to store cropped images
os.mkdir("test_cropped_images")

# read in labels csv
df = pd.read_csv("test_bounding_box.csv", index_col = False)
print(df.head())

# set working directory to folder with check images
os.getcwd()
os.chdir('/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/test')

# setting path of output folder
path = "/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/test_cropped_images" 

#Taking out -1 left value and replacing it with 0 
df[['x_left']] = df[['x_left']].clip(lower = 0)


# cropping
for i in range(len(df)):
  img = cv2.imread(df.filename[i]) # read in image
      # crop image using coordinates from df
  cropped_image = img[int(df.y_top[i]):int(df.y_bottom[i]), int(df.x_left[i]):int(df.x_bottom[i])] # img[ymin:ymax, xmin:xmax]

      # save cropped image to output folder
  cv2.imwrite(os.path.join(path , df.filename[i]), cropped_image)


############################################################

# Padding Cropped Images
# Code adapted from https://stackoverflow.com/a/59698237

# make folder to store padded images
os.chdir("/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3")
os.mkdir("test_images_padded")

# change working directory to cropped images folder
os.chdir("/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/test_cropped_images")

# setting path of output folder
path =  "/Users/marisamedina/Desktop/BZAN_554_Deep_Learning/assignment3/test_images_padded"


for i in range(len(df)):
    img = cv2.imread(df.filename[i]) # read in cropped image - can use same df and filnename column since names are the same, just in a different folder
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color (white) for padding
    new_image_width = int(max(df.width))
    new_image_height = int(max(df.height))
    color = (255,255,255)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
        x_center:x_center+old_image_width] = img

    """ # view result
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows() """

    # save result
    cv2.imwrite(os.path.join(path , df.filename[i]), result)


######################
### Project Phases ###
######################

# Read in data
# Padding
# Rescaling RGB values
# Code examples: image_dense.py, images_convnet.py (from Dropbox)
