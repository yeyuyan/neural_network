# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:57:17 2020

@author: YE677
"""

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#load dataset
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

seed = 7
np.random.seed(seed)

#create model
def create_model(optimizer = 'adam', init = 'glorot_uniform'):
    
    model = Sequential()
    model.add(Dense(units = 4, activation = 'relu', input_dim = 4, 
                    kernel_initializer = init))
    model.add(Dense(units = 6, activation = 'relu', kernel_initializer = init))
    model.add(Dense(units = 3, activation = 'softmax', kernel_initializer = init))
    
    model.compile(loss = 'categorical_crossentropy',optimizer = optimizer, 
                  metrics = ['accuracy'])
    
    return model

model = KerasClassifier(build_fn = create_model, epochs = 200, batch_size = 5, 
                       verbose = 0)


#evaluate model
kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)
results = cross_val_score(model, x, Y, cv = kfold)
print('Accuracy: %.2f%% (%.2f)' % (results.mean()*100, results.std()))
