# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:06:00 2020

@author: YE677
"""

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense,Flatten

seed = 7
np.random.seed(seed)

num_words = 5000
max_length = 500
out_dimension = 32
batch_size = 128
epochs = 2

# embedding
def create_model():
    
    model = Sequential()
    model.add(Embedding(num_words, out_dimension, input_length = max_length))
    model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', 
                     activation = 'relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    model.add(Dense(250, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                  metrics = ['accuracy'])
    print(model.summary())
    
    return model

if __name__=='__main__':
    
    (x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words 
    = num_words)
    x_train = sequence.pad_sequences(x_train, maxlen = max_length)
    x_validation = sequence.pad_sequences(x_validation, maxlen = max_length)
    
    model = create_model()
    model.fit(x_train, y_train, validation_data = (x_validation, y_validation),
              batch_size = batch_size, epochs = epochs, verbose = 2)
    
