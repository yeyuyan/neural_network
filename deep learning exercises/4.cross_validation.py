# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:42:00 2020

@author: YE677
"""


from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np

seed = 7
np.random.seed(seed)

# introduction of data
dataset = np.loadtxt('pima-indians-diabetes.csv',delimiter = ',')
#input x
x = dataset[:, 0 : 8]
# output Y
Y = dataset[:, 8]

# cross validation
kfold = StratifiedKFold(n_splits = 10, random_state = seed, shuffle = True)
cvscores = []


for train,validation in kfold.split(x,Y): #kfold.split gives the two sets of index numbers for training and validation for each split
    #definition of model
    model = Sequential()
    model.add(Dense(12, input_dim = 8, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    #compilation of model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics=['accuracy']) #metrics 衡量标准

    #trainning
    model.fit(x[train], Y[train], epochs = 150, batch_size = 10, verbose = 0)

    #evaluation
    scores = model.evaluate(x[validation], Y[validation], verbose = 0)
    print('\n%s : %.2f%%' % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1]*100)

#print average value and standard deviation
print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores), np.std(cvscores)))
