# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:42:00 2020

@author: YE677
"""


from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

seed = 7
np.random.seed(seed)

# introduction of data
dataset = np.loadtxt('pima-indians-diabetes.csv',delimiter = ',')
#input x
x = dataset[:, 0 : 8]
# output Y
Y = dataset[:, 8]

#split data for evaluation
x_train, x_validation, Y_train, Y_validation = train_test_split(x,Y,
test_size = 0.2,random_state = seed)

#definition of model
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#compilation of model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics=['accuracy']) #metrics 衡量标准

#trainning
#model.fit(x=x, y=Y, epochs = 150, batch_size = 10)
#model.fit(x=x, y=Y, epochs = 150, batch_size = 10, validation_split = 0.2) # use 20% Data to evaluate
model.fit(x_train, Y_train, validation_data = (x_validation, Y_validation),
          epochs = 150, batch_size = 10)#define the validation dataset

#evaluation
score = model.evaluate(x=x, y=Y)
print('\n%s : %.2f%%' % (model.metrics_names[1], score[1]*100))
