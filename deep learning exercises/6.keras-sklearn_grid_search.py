# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:42:00 2020

@author: YE677
"""


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


def create_model(optimizer = 'adam', init = 'glorot_uniform'):
    #definition of model
    model = Sequential()
    model.add(Dense(12, kernel_initializer = init, input_dim = 8, activation = 'relu'))
    model.add(Dense(8, kernel_initializer = init, activation = 'relu'))
    model.add(Dense(1, kernel_initializer = init, activation = 'sigmoid'))

    #compilation of model
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer,
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
model = KerasClassifier(build_fn = create_model, verbose = 0)

# define the params to adjust
param_grid = {}
param_grid['optimizer'] = ['rmsprop','adam']
param_grid['init'] = ['glorot_uniform', 'normal', 'uniform']
param_grid['epochs'] = [50, 100]
param_grid['batch_size'] = [5, 10]
#adjust params
grid = GridSearchCV(estimator = model, param_grid = param_grid)
results = grid.fit(x,Y)

#print the results
print('Best: %f using %s' % (results.best_score_, results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print('%f (%f) with %r' % (mean, std, param))

