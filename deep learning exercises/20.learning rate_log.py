# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:57:17 2020

@author: YE677
"""

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from math import pow, floor
from keras.callbacks import LearningRateScheduler

seed = 7
np.random.seed(seed)

#load dataset
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

#calculate learning rate using log decay
def step_decay(epoch):
    init_rate = 0.1
    drop = 0.5
    epoch_drop = 10
    lrate = init_rate*pow(drop,floor((1+epoch)/epoch_drop))
    return lrate

#create model
def create_model(init = 'glorot_uniform'):
    
    model = Sequential()
    model.add(Dense(units = 4, activation = 'relu',input_dim = 4, kernel_initializer = init))
    model.add(Dense(units = 6, activation = 'relu', kernel_initializer = init))
    model.add(Dense(units = 3, activation = 'softmax', kernel_initializer = init))
    
    #optimize model
    learningRate = 0.1
    momentum = 0.9
    decay_rate = 0.0
    sgd = SGD(lr = learningRate, momentum = momentum, decay = decay_rate, 
              nesterov = False)
    
    model.compile(loss = 'categorical_crossentropy',optimizer = sgd, 
                  metrics = ['accuracy'])
    
    return model

#learning rate drop out callbacks
lrate = LearningRateScheduler(step_decay)
model = KerasClassifier(build_fn = create_model, epochs = 200, batch_size = 5, 
                       verbose = 1,callbacks = [lrate])
model.fit(x,Y)