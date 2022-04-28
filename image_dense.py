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

#Create architecture
inputs = tf.keras.layers.Input(shape=(28,28), name='input') 
flatten = tf.keras.layers.Flatten(name='flatten')(inputs)
hidden1 = tf.keras.layers.Dense(300, activation = 'relu',name='hidden1')(flatten)
hidden2 = tf.keras.layers.Dense(100, activation = 'relu',name='hidden2')(hidden1)
outputs = tf.keras.layers.Dense(10, activation = 'softmax',name='out')(hidden2)

model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.summary()
#Compile model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))

#Fit model
model.fit(x=X_train,y=y_train, batch_size=1, epochs=2) 


#Compute multiclass accuray
yhat = model.predict(x=X_test)
yhat_sparse = [int(np.where(yhat_sub ==np.max(yhat_sub))[0]) for yhat_sub in yhat]
y_test
sum(yhat_sparse == y_test) / len(y_test)
