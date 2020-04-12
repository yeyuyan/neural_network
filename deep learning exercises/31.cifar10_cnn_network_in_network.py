# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:09:58 2020

@author: YE677
"""
import keras
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, GlobalAveragePooling2D
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler, TensorBoard

batch_size = 128
epochs = 200
iterations = 391
num_classes = 10
dropout = 0.5
log_filepath = './nin'


def normalize_preprocessing(x_train, x_validation):
    x_train = x_train.astype('float32')
    x_validation = x_validation.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.0032, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] -mean[i])/std[i]
        x_validation[:,:,:,i] = (x_validation[:,:,:,i] -mean[i])/std[i]
        
    return x_train, x_validation

def scheduler(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <= 120:
        return 0.01
    if epoch <= 160:
        return 0.002
    return 0.0004
    
def create_model():
    
    model = Sequential()
    
    model.add(Conv2D(192, (5,5), padding = 'same',
                     kernel_regularizer = keras.regularizers.l2(0.0001),
                     kernel_initializer = RandomNormal(stddev = 0.01),
                     input_shape = x_train.shape[1:],
                     activation = 'relu'))
    
    model.add(Conv2D(160, (1,1), padding = 'same',
                     kernel_regularizer = keras.regularizers.l2(0.0001),
                     kernel_initializer = RandomNormal(stddev = 0.05),
                     activation = 'relu'))
    
    model.add(Conv2D(96, (1,1), padding = 'same',
                     kernel_regularizer = keras.regularizers.l2(0.0001),
                     kernel_initializer = RandomNormal(stddev = 0.05),
                     activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size = (3,3), stride = (2,2), padding = 'same'))
    
    model.add(Dropout(dropout))
    
    model.add(Conv2D(192, (5,5), padding = 'same',
                     kernel_regularizer = keras.regularizers.l2(0.0001),
                     kernel_initializer = RandomNormal(stddev = 0.05),
                     activation = 'relu'))
    
    model.add(Conv2D(192, (1,1), padding = 'same',
                     kernel_regularizer = keras.regularizers.l2(0.0001),
                     kernel_initializer = RandomNormal(stddev = 0.05),
                     activation = 'relu'))
     
    model.add(Conv2D(192, (1,1), padding = 'same',
                     kernel_regularizer = keras.regularizers.l2(0.0001),
                     kernel_initializer = RandomNormal(stddev = 0.05),
                     activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size = (3,3), stride = (2,2), padding = 'same'))
    
    model.add(Dropout(dropout))
    
    model.add(Conv2D(192, (3,3), padding = 'same',
                     kernel_regularizer = keras.regularizers.l2(0.0001),
                     kernel_initializer = RandomNormal(stddev = 0.05),
                     activation = 'relu'))
    
    model.add(Conv2D(192, (1,1), padding = 'same',
                     kernel_regularizer = keras.regularizers.l2(0.0001),
                     kernel_initializer = RandomNormal(stddev = 0.05),
                     activation = 'relu'))
    
    model.add(Conv2D(10, (1,1), padding = 'same',
                     kernel_regularizer = keras.regularizers.l2(0.0001),
                     kernel_initializer = RandomNormal(stddev = 0.05),
                     activation = 'relu'))
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Activation('softmax'))
    
    sgd = SGD(lr = 0.1, momentum = 0.9, nesterov = True)
    model.compile(loss = 'categorical_crossentropy',optimizer = sgd, 
                  metrics = ['accuracy'])
    
    return model

if __name__=='__main__':
    np.random.seed(seed = 7)
    
    #import data
    (x_train, y_train), (x_validation, y_validation) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_validation = np_utils.to_categorical(y_validation, num_classes)
    x_train, x_validation = normalize_preprocessing(x_train, x_validation)

#build model
model = create_model()
print(model.summary())

#set callbacks 
tb_cb = TensorBoard(log_dir = log_filepath, histogram_freq = 0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr, tb_cb]

model.fit(x=x_train, y=y_train, epochs=epochs, batch_size = batch_size, 
          callbacks = cbks ,validation_data = (x_validation, y_validation), 
          verbose = 2)
model.save('nin.h5')