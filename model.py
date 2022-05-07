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
path_repo_matt = "C:/MSBA/22SP/BZAN 554 Deep Learning/Projects/project3/bzan554-cnn-svhn"
path_repo_jake='C:/Users/jamfo/Documents/Deep Learning/bzan554-cnn-svhn' #path to local repo

train_images_path_jake='C:/Users/jamfo/Documents/Deep Learning/SVHN/train_images_grey'
test_images_path_jake='C:/Users/jamfo/Documents/Deep Learning/SVHN/test_images_grey'

train_images_path_matt = "C:/MSBA/22SP/BZAN 554 Deep Learning/Projects/project3/SVHN/train_images_grey"
test_images_path_matt = "C:/MSBA/22SP/BZAN 554 Deep Learning/Projects/project3/SVHN/test_images_grey"

repo_path = path_repo_matt
train_images_path = train_images_path_matt
test_images_path = test_images_path_matt

os.chdir(repo_path)

#create y train and y test
train_df=pd.read_csv('train_labels.csv')
test_df=pd.read_csv('test_labels.csv')
y_train = train_df[['label']].to_numpy(dtype=np.float32)
y_test = test_df[['label']].to_numpy(dtype=np.float32)


#create X train
os.chdir(train_images_path)

X_train=[]

for i in range(33402):
    img=mpimg.imread(str(i+1)+'.png')#.astype(np.float32)
    X_train.append(img)
X_train = np.asarray(X_train)

# plt.imshow(X_train[33401])
# plt.show()
# y_train[33401]


#create X test
os.chdir(test_images_path)

X_test=[]

for i in range(13068):
    img=mpimg.imread(str(i+1)+'.png')#.astype(np.float32)
    X_test.append(img)
X_test = np.asarray(X_test)

# plt.imshow(X_test[13067])
# plt.show()
# y_test[13067]

###################### CNN MODEL ######################
np.unique(y_train.astype(np.uint8)) # 256 unique classes
X_train[1].shape # each image is 403x434
n_softmax = len(str(np.max(y_train))) + 1 # 6+1 softmax layers

##### Model architecture #####
inputs = tf.keras.layers.Input(shape=(403,434,1), name="input")
x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding="same", activation="relu")(inputs)

# MaxPooling2D: pool_size is window size over which to take the max
x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=31, padding="valid")(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=7, strides=1, padding="same", activation="relu")(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=7, strides=1, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=31, padding="valid")(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=7, strides=1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=7, strides=1, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=31, padding="valid")(x)

# dense layers expect 1D array of features for each instance so we need to flatten.
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
yhat = tf.keras.layers.Dense(n_softmax, activation='softmax')(x)

model = tf.keras.Model(inputs = inputs, outputs = yhat)
model.summary()

# Compile model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001))

# Fit model
model.fit(x=X_train, y=y_train, batch_size=1, epochs=2) 

# Compute multiclass accuray
yhat = model.predict(x=X_test)
yhat_sparse = [int(np.where(yhat_sub ==np.max(yhat_sub))[0]) for yhat_sub in yhat]
y_test
sum(yhat_sparse == y_test) / len(y_test)

