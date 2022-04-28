# MNIST dataset
# 70,000 small images of handwritten digits 
# each images is labeled with the digit it represents

#download the dataset:
#If you are on a mac, might need to run this in terminal:
"/Applications/Python 3.7/Install Certificates.command"
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test,y_test) = mnist.load_data()

#How many images in train and their size?
X_train.shape
# Images: 60000
# Size: 28 x 28 pixels
#We will treat every pixel as an input feature.
#Each feature represents the pixel's intensity from
#white (0) to black (255)
#That means we have 784 features (because 28 x 28 pixels)


#Example:

import matplotlib.pyplot as plt
first_image = X_train[0]
first_image.shape
plt.imshow(first_image, cmap = 'binary')
plt.show()
#looks like a 5
#let's verify the label:
y_train[0]


#scaling the features:
X_train = X_train / 255.0
X_test = X_test / 255.0


#Let's build a multi-class classifier
#Unique classes?
import numpy as np
np.unique(y_train.astype(np.uint8))
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
 
#Specify architecture

#As we go deeper toward the output it makes sense to increase the number of filters
#since the number of low-level features is often low (circles, lines, ...), but there are many
#different ways to combine low-level features into higher-level features (nose, ear, ...).
#It is common to double the number of filters after each pooling layer (e.g., 64, 128, 256) as the pooling layer 
#divides each spatial dimension by 2. This prevents growth in the computational load.

#shape is 28,28,1 because grayscale (single color channel)
#Conv layers requires 4 dimensional input (i.e., if we input three dimensions and add batch dim we get 4D)
inputs = tf.keras.layers.Input(shape=(28,28,1), name='input') 
#Conv2D layer (2D does not refer to gray scale (a PET scan would be 3D))
x = tf.keras.layers.Conv2D(filters=64,kernel_size = 7, strides = 1, padding = "same", activation = "relu")(inputs)
#MaxPooling2D: pool_size is window size over which to take the max
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
#dense layers expect 1D array of features for each instance so we need to flatten.
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation = 'relu')(x)
x = tf.keras.layers.Dense(64, activation = 'relu')(x)
yhat = tf.keras.layers.Dense(10, activation = 'softmax')(x)

#Why do we stack two convolutional layers followed by a pooling layer, as opposed to having each convolutional layer followed by a pooling layer?
# Answer: every convolutional layer creates a number of feature maps (e.g,64) that are individually connected to the previous layer.
# By stacking two convolutional layers before inserting a pooling layer we allow the second convolutional layer to learn from the noisy signal, as opposed to the clean signal.

model = tf.keras.Model(inputs = inputs, outputs = yhat)
model.summary()
#Compile model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))

#Fit model
model.fit(x=X_train,y=y_train, batch_size=32, epochs=1) 


#Compute multiclass accuray
yhat = model.predict(x=X_test)
yhat_sparse = [int(np.where(yhat_sub ==np.max(yhat_sub))[0]) for yhat_sub in yhat]
y_test
sum(yhat_sparse == y_test) / len(y_test)
