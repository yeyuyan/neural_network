# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:37:46 2020

@author: YE677
"""
from nltk import word_tokenize
from gensim import corpora
from pyecharts.charts import WordCloud
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
import numpy as np
from keras.utils import np_utils

filename = '46.Alice/Alice.txt'
dict_file = '46.Alice/dict_file.txt'
model_json_file = '46.Alice/simple_lstm.json'
model_hd5_file = '46.Alice/simple_lstm.hd5'
batch_size = 128
epochs = 300
max_len = 20
dict_len = 2788
document_max_len = 33200


def load_dataset():
    with open(file = filename, mode = 'r') as file:
        document = []
        lines = file.readlines()
        for line in lines:
            value = clear_data(line)
            if value != '':
                #seperate words of one line
                for s in word_tokenize(value):
                    if s=='CHAPTER':
                        break
                    else:
                        document.append(s.lower())
    return document
            
            
def clear_data(s):
    value = s.replace('\ufeff', '').replace('\n','')
    return value

def word_to_integer(document):
    dic = corpora.Dictionary([document])
    dic.save_as_text(dict_file)
    dic_set = dic.token2id
    values = []
    for word in document:
        values.append(dic_set[word])
    return values

def show_word_cloud(document):
    left_words = ['.',',''?','!',';',':','\'','(',')']
    dic = corpora.Dictionary([document])
    words_set = dic.doc2bow(document)
    
    words, frequences = [], []
    for item in words_set:
        key = item[0]
        frequence = item[1]
        word = dic.get(key = key)
        if word not in left_words:
            words.append(word)
            frequences.append(frequence)
    #word_cloud = WordCloud(width = 1000, height = 620)
    word_cloud = WordCloud()
    word_cloud.add("",list(zip(words, frequences)), shape = 'circle', 
                   word_size_range = [20,100])
    word_cloud.render('46.Alice/words.html')
    
def create_model():
    model = Sequential()
    model.add(Embedding(input_dim = dict_len, output_dim = 32, input_length = max_len))
    model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same',
                     activation = 'relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(units = dict_len, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    model.summary()
    return model
    
def make_x(document):
    dataset = make_dataset(document)
    x = dataset[0:dataset.shape[0] -1, :]
    return x

def make_y(document):
    dataset = make_dataset(document)
    y = dataset[1: dataset.shape[0], 0]
    return y

#seperate doc using a fix length
def make_dataset(document):
    dataset = np.array(document[0:document_max_len])
    dataset = dataset.reshape(int(document_max_len/max_len),max_len)
    return dataset


    
if __name__=='__main__':
    document = load_dataset()
    show_word_cloud(document)
    
    #transform words into integers
    values = word_to_integer(document)
    x_train = make_x(values)
    y_train = make_y(values)
    #one shot encoder
    y_train = np_utils.to_categorical(y_train, dict_len)
    
    model = create_model()
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, 
              verbose = 2)
    
    model_json = model.to_json()
    with open(model_json_file, 'w') as file:
        file.write(model_json)
    model.save_weights(model_hd5_file)