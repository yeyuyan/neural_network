# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:57:17 2020

@author: YE677
"""

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


seed = 7
np.random.seed(seed)

#load dataset
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

#transform label into number
Y_labels = to_categorical(Y,num_classes = 3)


#create model
def create_model(optimizer = 'rmsprop', init = 'glorot_uniform'):
    
    model = Sequential()
    model.add(Dense(units = 4, activation = 'relu', input_dim = 4, 
                    kernel_initializer = init))
    model.add(Dense(units = 6, activation = 'relu', kernel_initializer = init))
    model.add(Dense(units = 3, activation = 'softmax', kernel_initializer = init))
    
    #load weights
    filepath = 'weights-improvement/weights.best.h5'
    model.load_weights(filepath = filepath)
    
    model.compile(loss = 'categorical_crossentropy',optimizer = optimizer, 
                  metrics = ['accuracy'])
    
    return model

model = create_model()

scores = model.evaluate(x,Y_labels,verbose = 0)
print('%s: %.2f%%' % (model.metrics_names[1],scores[1]*100))
