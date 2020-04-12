# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:42:00 2020

@author: YE677
"""


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier


def create_model():
    #definition of model
    model = Sequential()
    model.add(Dense(12, input_dim = 8, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    #compilation of model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                  metrics=['accuracy']) #metrics 衡量标准
    
    return model

seed = 7
np.random.seed(seed)

# introduction of data
dataset = np.loadtxt('pima-indians-diabetes.csv',delimiter = ',')
#input x
x = dataset[:, 0 : 8]
# output Y
Y = dataset[:, 8]

# create model for scikit-learn
model = KerasClassifier(build_fn = create_model, epochs = 150, batch_size = 10,
                        verbose = 0)

# 10 cross validation
#instantiation of un object de class StratifiedKFold 
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)
result = cross_val_score(model, x, Y, cv = kfold)
print(result.mean())

