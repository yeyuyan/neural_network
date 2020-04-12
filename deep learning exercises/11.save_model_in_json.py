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
from keras.models import model_from_json

#load dataset
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

#transform label into number
Y_labels = to_categorical(Y,num_classes = 3)

seed = 7
np.random.seed(seed)

#create model
def create_model(optimizer = 'rmsprop', init = 'glorot_uniform'):
    
    model = Sequential()
    model.add(Dense(units = 4, activation = 'relu', input_dim = 4, 
                    kernel_initializer = init))
    model.add(Dense(units = 6, activation = 'relu', kernel_initializer = init))
    model.add(Dense(units = 3, activation = 'softmax', kernel_initializer = init))
    
    model.compile(loss = 'categorical_crossentropy',optimizer = optimizer, 
                  metrics = ['accuracy'])
    
    return model

model = create_model()
model.fit(x,Y_labels,epochs = 200,batch_size = 5,verbose = 0)
scores = model.evaluate(x,Y_labels,verbose = 0)
print('%s %.2f%%' % (model.metrics_names[1],scores[1]*100))

#save model as json doc
model_json = model.to_json()
with open('model.json','w') as file:
    file.write(model_json)

#save model's weights
model.save_weights('model.json.h5')

#import model from json doc
with open('model.json','r') as file:
    model_json = file.read()

new_model = model_from_json(model_json)
new_model.load_weights('model.json.h5')

new_model.compile(loss = 'categorical_crossentropy',optimizer = 'rmsprop', 
                  metrics = ['accuracy'])
scores = new_model.evaluate(x,Y_labels,verbose = 0)
print('%s %.2f%%' % (model.metrics_names[1],scores[1]*100))