# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:50:34 2020

@author: YE677
"""

from sklearn import datasets 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#import dataset
dataset = datasets.load_boston()

x = dataset.data
Y = dataset.target

#set random number seed
seed = 7
np.random.seed(seed)

#create model
def create_model(units_list = [13], optimizer = 'adam', init = 'normal'):
    
    model = Sequential()
    units = units_list[0]
    model.add(Dense(units=units, activation = 'relu' ,input_dim = 13,
                    kernel_initializer = init))
    
    for units in units_list[1:]:
        model.add(Dense(units = units, activation = 'relu', 
                        kernel_initializer = init))
    model.add(Dense(units = 1, activation = 'relu', kernel_initializer = init))
    
    model.compile(loss = 'mean_squared_error',optimizer = optimizer)
    
    return model

model = KerasRegressor(build_fn = create_model, epochs = 200, batch_size = 5, 
                       verbose = 0)

#srandardize data

steps=[]
steps.append(('standardize',StandardScaler()))
steps.append(('mlp',model))
pipeline = Pipeline(steps)



#evaluate the model
kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)
#results = cross_val_score(pipeline, x, Y, cv = kfold)
#print('Standardize: %.2f (%.2f) MSE' % (results.mean(), results.std()))
results = cross_val_score(model, x, Y, cv = kfold)
print('Baseline: %.2f (%.2f) MSE' % (results.mean(), results.std()))