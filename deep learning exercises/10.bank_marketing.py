# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:05:09 2020

@author: YE677
"""

import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


#introduce data et transform data into numbers
dataset = read_csv('bank/bank.csv',delimiter = ';')

dataset['job'] = dataset['job'].replace(to_replace=['admin.','unknown',
       'unemployed','management','housemaid','entrepreneur','student',
       'blue-collar','self-employed','retired','technician','services'],
value = [0,1,2,3,4,5,6,7,8,9,10,11])

dataset['marital'] = dataset['marital'].replace(to_replace = ['married','single',
       'divorced'], value = [0,1,2])

dataset['education'] = dataset['education'].replace(to_replace = ['unknown',
       'secondary','primary','tertiary'], value = [0,1,2,3])

dataset['default'] = dataset['default'].replace(to_replace = ['no','yes'],value
       = [0,1])

dataset['housing'] = dataset['housing'].replace(to_replace = ['no','yes'],value
       = [0,1])

dataset['loan'] = dataset['loan'].replace(to_replace = ['no','yes'],value
       = [0,1])

dataset['contact'] = dataset['contact'].replace(to_replace = ['cellular',
       'unknown','telephone'],value = [0,1,2])

dataset['poutcome'] = dataset['poutcome'].replace(to_replace = ['unknown',
       'other','success','failure'],value = [0,1,2,3])

dataset['month'] = dataset['month'].replace(to_replace = ['jan','feb','mar',
       'apr','may','jun','jul','aug','sep','oct','nov','dec'], value = [i for i
                                                            in range(1,13)])

dataset['y'] = dataset['y'].replace(to_replace = ['no','yes'], value = [0,1])


#seperate input and output
array = dataset.values
x = array[:, 0:16]
Y = array[:, 16]

#set random seed
seed = 7
np.random.seed(seed)

def create_model(units_list = [16], optimizer = 'adam', init = 'normal'):
    model = Sequential()
    
    units = units_list[0]
    model.add(Dense(units = units, activation = 'relu',input_dim = 16, 
                    kernel_initializer = init))
    
    for units in units_list[1:]:
        model.add(Dense(units = units, activation = 'relu',input_dim = 16, 
                        kenel_initializer = init))
    
    model.add(Dense(units = 1, activation = 'sigmoid',kernel_initializer = init))
    
    model.compile(loss = 'binary_crossentropy',optimizer = optimizer, 
                  metrics = ['accuracy'])
    
    return model

model = KerasClassifier(build_fn = create_model, epochs = 200, batch_size = 5,
                        verbose = 0)

#Standardize data
new_x = StandardScaler().fit_transform(x)
kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)
#results = cross_val_score(model,x,Y,cv = kfold)
results = cross_val_score(model,new_x,Y,cv = kfold)
print('Accurary: %.2f%% (%.2f)' % (results.mean()*100, results.std()))
