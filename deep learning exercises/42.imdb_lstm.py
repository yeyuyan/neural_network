# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:36:26 2020

@author: YE677
"""

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout

seed = 7
np.random.seed(seed)

num_words = 5000
max_length = 500
out_dimension = 32
batch_size = 128
epochs = 2
dropout_rate = 0.2


# embedding
def create_model():
    
    model = Sequential()
    model.add(Embedding(num_words, out_dimension, input_length = max_length))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units = 100))
    model.add(Dropout(dropout_rate))
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
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, 
              verbose = 2)
    scores = model.evaluate(x_validation, y_validation, verbose = 2)
    print('Accuracy: %.2f%%' % (scores[1]*100))
    




