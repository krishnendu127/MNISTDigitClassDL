# -*- coding: utf-8 -*-
"""MNISTDigitClass.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dKvbVuEGsbjb311AolPWjW_ERA5at-AP
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

"""Loading the mnist data from keras.dataset

"""

(x_train,y_train),(x_test, y_test) = mnist.load_data()

type(x_train)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

"""Training Data= 60000
Test Data=10000
Image dimension= 28*28
Grayscale image
"""

print(x_train[10].shape)

#converting image from numpy array to displayable

plt.imshow(x_train[10])
plt.show()

print(y_train[10])

"""both number matches

image labels-
"""

print(y_train.shape , y_test.shape)

#unique values in y_train
print(np.unique(y_train))

print(np.unique(y_test))

"""we can use these labels as such or we can also apply one hot encoding"""

#scaling the values

x_train=x_train/255
x_test=x_test/255

print(x_train[10])

"""building a neural network"""

#setting up the layers of my neural network
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

#compiling the model
model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

#training the network

model.fit(x_train, y_train, epochs=10)

"""training data accuracy=98.79%)"""

# accuracy on test data:

loss, accuracy=model.evaluate(x_test,y_test)

"""test data accuracy= 97% accurate"""

print(x_test.shape)

#first data point in x_test

plt.imshow(x_test[0])
plt.show()

print(y_test[0])

y_pred= model.predict(x_test)

print(y_pred.shape)

print(y_pred[0])

"""model.predict gives the prediction probability of each class for that datapoint"""

#converting the prediction probabilities to class label

label_for_first_test_image=np.argmax(y_pred[0])
print(label_for_first_test_image)

#converting the prediction probabilities to class label for all test data points
y_pred_labels= [np.argmax(i) for i in y_pred]
print(y_pred_labels)

print(y_test)

"""using confusion matrix"""

conf_mat= confusion_matrix(y_test, y_pred_labels)

print(conf_mat)

plt.figure(figsize=(15,7))
sns.heatmap(conf_mat, annot =True , fmt='d', cmap='Blues')
plt.ylabel("True labels")
plt.xlabel("predicted labels")

"""best accuracy for 1 image as seen from above

Now, building a predictive system
"""

input_image_path='/content/mnist.jpeg'

input_img=cv2.imread(input_image_path)

type(input_img)

cv2_imshow(input_img)

input_img.shape

grayscale = cv2.cvtColor(input_img , cv2.COLOR_RGB2GRAY)

grayscale.shape

input_img=grayscale

input_img_resized=cv2.resize(input_img, (28,28))

input_img_resized.shape

cv2_imshow(input_img_resized)

input_img_resized=input_img_resized/255

"""converting it to a single array so that it can be fed to the model"""

image_reshaped= np.reshape(input_img_resized, [1,28,28])

input_pred= model.predict(image_reshaped)
print(input_pred)

input_pred_label= np.argmax(input_pred)
print(input_pred_label)

