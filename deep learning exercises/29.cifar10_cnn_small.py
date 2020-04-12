# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:09:58 2020

@author: YE677
"""
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras import backend
backend.set_image_data_format('channels_first')

#set random seed
seed = 7
np.random.seed(seed)

#import data
(X_train, y_train), (X_validation, y_validation) = cifar10.load_data()

#standardization
X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
X_train = X_train / 255.0
X_validation = X_validation / 255.0

#one shot
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)
num_classes = y_train.shape[1]

def create_model(epochs = 25):
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape = (3,32,32), padding = 'same',
                     activation = 'relu', kernel_constraint = maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3,3), padding = 'same', activation = 'relu', 
                     kernel_constraint = maxnorm(3)))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu', kernel_constraint = maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr = lrate, momentum = 0.9, decay = decay, nesterov = False)
    model.compile(loss = 'categorical_crossentropy',optimizer = sgd, 
                  metrics = ['accuracy'])
    return model

epochs = 25
model = create_model(epochs)
model.fit(x=X_train, y=y_train, epochs=epochs, batch_size = 32, verbose = 2)
score = model.evaluate(X_validation, y_validation, verbose = 0)
print('Accuracy: %.2f%% ' % (score[1]*100))