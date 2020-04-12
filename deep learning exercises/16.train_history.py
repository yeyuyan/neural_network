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
from matplotlib import pyplot as plt

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
    
    model.compile(loss = 'categorical_crossentropy',optimizer = optimizer, 
                  metrics = ['accuracy'])
    
    return model

model = create_model()

history = model.fit(x, Y_labels, validation_split = 0.2, epochs = 200, 
                    batch_size = 5, verbose = 0)

scores = model.evaluate(x,Y_labels,verbose = 0)
print('%s: %.2f%%' % (model.metrics_names[1],scores[1]*100))

#history list
print(history.history.keys())

#history of accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()

#history of loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.show()