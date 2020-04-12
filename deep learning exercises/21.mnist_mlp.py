# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:35:55 2020

@author: YE677
"""

from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

#import mnist dataset from keras
(X_train, y_train), (X_validation, y_validation) = mnist.load_data()

#show first 4 pictures
plt.subplot(221)
plt.imshow(X_train[0],cmap = plt.get_cmap('gray'))

plt.subplot(222)
plt.imshow(X_train[1],cmap = plt.get_cmap('gray'))

plt.subplot(223)
plt.imshow(X_train[2],cmap = plt.get_cmap('gray'))

plt.subplot(224)
plt.imshow(X_train[3],cmap = plt.get_cmap('gray'))

plt.show()

#set random seed
seed = 7
np.random.seed(seed)

num_pixels = X_train.shape[1] * X_train.shape[2]
print(num_pixels)
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_validation = X_validation.reshape(X_validation.shape[0], num_pixels).astype('float32')

#normalize data to 0-1
X_train = X_train/255
X_valdiation = X_validation/255

#one-hot 
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_validation.shape[1]
print(num_classes)

#define multilayer perceptron model
def create_model():
    model = Sequential()
    model.add(Dense(units = num_pixels, input_dim = num_pixels, 
                    kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(units = num_classes, input_dim = num_classes, 
                    kernel_initializer = 'normal', activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                  metrics = ['accuracy'])
    
    return model

model = create_model()
model.fit(X_train,y_train,epochs = 10, batch_size = 200)

score = model.evaluate(X_validation,y_validation)
print('MLP: %.2f%%' % (score[1]*100))
