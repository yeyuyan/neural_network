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
    
    model.compile(loss = 'mean_squared_error', optimizer = optimizer)
    
    return model

model = KerasRegressor(build_fn = create_model, epochs = 200, batch_size = 5, 
                       verbose = 0)


#choose params to adjust
param_grid = {}
param_grid['units_list'] = [[20], [13, 6]]
param_grid['optimizer'] = ['adam','rmsprop']
param_grid['init'] = ['glorot_uniform', 'normal']
param_grid['epochs'] = [100,200]
param_grid['batch_size'] = [5,20]

#adjust params
scaler = StandardScaler()
scaler_x = scaler.fit_transform(x)
grid = GridSearchCV(estimator = model, param_grid = param_grid)
results = grid.fit(scaler_x, Y)

#print results
print('Best: %f using %s' % (results.best_score_, results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']

for mean, std, param in zip(means,stds,params):
    print('%f (%f) with %r' % (mean,std,param))
