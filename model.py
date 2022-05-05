import tensorflow as tf
import numpy as np
import pandas as pd
import os
from fileinput import filename
from os import listdir, rename
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

marisa_path_repo=''
cole_path_repo=''
path_repo_jake='C:/Users/jamfo/Documents/Deep Learning/bzan554-cnn-svhn' #path to local repo

train_images_path_jake='C:/Users/jamfo/Documents/Deep Learning/SVHN/train_images_grey'
test_images_path_jake='C:/Users/jamfo/Documents/Deep Learning/SVHN/test_images_grey'

repo_path=path_repo_jake
train_images_path=train_images_path_jake
test_images_path=test_images_path_jake

os.chdir(repo_path)

#create y train and y test
train_df=pd.read_csv('train_labels.csv')
test_df=pd.read_csv('test_labels.csv')
y_train = train_df[['label']].to_numpy()
y_test = test_df[['label']].to_numpy()


#create X train
os.chdir(train_images_path)

X_train=[]

for i in range(33402):
    img=mpimg.imread(str(i+1)+'.png')
    X_train.append(img)

plt.imshow(X_train[33401])
plt.show()
y_train[33401]


#create X test
os.chdir(test_images_path)

X_test=[]

for i in range(13068):
    img=mpimg.imread(str(i+1)+'.png')
    X_test.append(img)

plt.imshow(X_test[13067])
plt.show()
y_test[13067]

