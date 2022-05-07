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

# plt.imshow(X_train[33401])
# plt.show()
# y_train[33401]


#create X test
os.chdir(test_images_path)

X_test=[]

for i in range(13068):
    img=mpimg.imread(str(i+1)+'.png')#.astype(np.float32)
    X_test.append(img)

# plt.imshow(X_test[13067])
# plt.show()
# y_test[13067]

###################### CNN MODEL ######################
np.unique(y_train.astype(np.uint8)) # 256 unique classes
X_train[1].shape # each image is 403x434
n_softmax = len(str(np.max(y_train))) + 1 # 6+1 softmax layers

##### Model architecture #####
def build_model(hp):
    # create model object
    model = tf.keras.Sequential([
    #adding first convolutional layer    
    tf.keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_1_filter', min_value=32, max_value=512, step=16),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_1_kernel', values = [1,16]),
        padding="same",
        strides = 1,
        #activation function
        input_shape=(403,434,1)),
    #adding first pooling layer   
    tf.keras.layers.MaxPooling2D(
        #adding filter 
        pool_size=hp.Int('pool_1_pool_size', min_value=1, max_value=10, step=16),
        # adding stride size
        strides=hp.Choice('pool_1_stride_size', values = [1,16]),
        padding="valid"),
    #adding 2nd convolutional layer    
    tf.keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_2_filter', min_value=32, max_value=512, step=16),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_2_kernel', values = [1,16]),
        padding="same",
        strides = 1,
        #activation function
        activation='relu'),
    #adding 3rd convolutional layer    
    tf.keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_3_filter', min_value=32, max_value=512, step=16),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_3_kernel', values = [1,16]),
        padding="same",
        strides = 1,
        #activation function
        activation='relu'),
    #adding 2nd POOLING layer   
    tf.keras.layers.MaxPooling2D(
        #adding filter 
        pool_size=hp.Int('pool_2_pool_size', min_value=1, max_value=10, step=16),
        # adding stride size
        strides=hp.Choice('pool_2_stride_size', values = [1,16]),
        padding="valid"),
        
    #adding 4th convolutional layer    
    tf.keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_4_filter', min_value=32, max_value=512, step=16),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_4_kernel', values = [1,16]),
        padding="same",
        strides = 1,
        #activation function
        activation='relu'),
    #adding 5th convolutional layer    
    tf.keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_5_filter', min_value=32, max_value=512, step=16),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_5_kernel', values = [1,16]),
        padding="same",
        strides = 1,
        #activation function
        activation='relu'),
    #adding 3rd POOLING layer   
    tf.keras.layers.MaxPooling2D(
        #adding filter 
        pool_size=hp.Int('pool_3_pool_size', min_value=1, max_value=10, step=16),
        # adding stride size
        strides=hp.Choice('pool_1_stride_size', values = [1,16]),
        padding="valid"),
        #activation function
    
    # adding flatten layer    
    tf.keras.layers.Flatten(),
    # adding dense layer    
    tf.keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=512, step=16),
        activation='relu'
    ),
    # adding dense layer    
    tf.keras.layers.Dense(
        units=hp.Int('dense_2_units', min_value=32, max_value=512, step=16),
        activation='relu'
    ),
    # output layer    
    tf.keras.layers.Dense(n_softmax, activation='softmax')
    ])
    #compilation of model
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[.001, .0001])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

#use fthis to hypertune the model before fitting it
from keras_tuner import RandomSearch

# creating randomsearch object
tuner = RandomSearch(build_model,
                    objective='val_accuracy',
                    max_trials = 5)

# search best parameter
# train_df = train_df.astype({"label": np.float32}, errors='raise') # force labels from 64-bit integer to 32-bit float
# test_df = test_df.astype({"label": np.float32}, errors='raise') # force labels from 64-bit integer to 32-bit float
train_df = train_df["label"].astype(np.float32)
test_df = test_df["label"].astype(np.float32)
tuner.search(train_df, y_train, epochs=3, validation_data=(test_df, y_test))

# get the best model 
model = tuner.get_best_models(num_models=1)[0]

#summary of best model
model.summary()


#Fit the best model from the hp search
model.fit(x=X_train, y=y_train, batch_size=1, epochs=2) 

#Compute multiclass accuray
yhat = model.predict(x=X_test)
yhat_sparse = [int(np.where(yhat_sub ==np.max(yhat_sub))[0]) for yhat_sub in yhat]
y_test
sum(yhat_sparse == y_test) / len(y_test)

