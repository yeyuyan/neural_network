# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:04:43 2020

@author: YE677
"""

from pandas import read_csv, DataFrame,concat
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense

batch_size = 72
epochs = 50
n_input = 2
n_train_hours = 365*24*4
n_validation_hours = 24*5
filename = 'pollution_original.csv'

def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

def load_dataset():
    dataset = read_csv(filename, parse_dates = [['year', 'month', 'day', 
                     'hour']], index_col = 0, date_parser = parse)
    
    #delete col 'no'
    dataset.drop('No', axis = 1, inplace = True)
    
    #set col name
    dataset.columns = ['pollution','dew','temp','press','wnd_dir','wnd_spd',
                       'snow','rain']
    dataset.index.name = 'date'
    
    #fill nan value
    dataset['pollution'].fillna(dataset['pollution'].mean(),inplace = True)
    
    return dataset

def convert_dataset(data, n_input = 1, out_index = 0, dropnan = True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [],[]
    #input sequences (t-n,...,t-1)
    for i in range(n_input,0,-1):
        cols.append(df.shift(i))#shift 把数据往时间轴推后
        names +=[('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    #print(cols)
    #output sequence t
    cols.append(df[df.columns[out_index]])
    names+=['result']
    #concatenate input and output
    result = concat(cols,axis = 1)
    result.columns = names
    #delete rows having nan value
    if dropnan:
        result.dropna(inplace = True)
    return result

#encode a list or a column of a index        
def class_encode(data, class_indexs):
    encoder = LabelEncoder()
    class_indexs = class_indexs if type(class_indexs) is list else [class_indexs]
    values = DataFrame(data).values
    for index in class_indexs:
        values[:, index] = encoder.fit_transform(values[:,index])
    
    return DataFrame(values) if type(data) is DataFrame else values

def create_model(lstm_input_shape):
    model = Sequential()
    model.add(LSTM(units = 50, input_shape = lstm_input_shape,
                   return_sequences = True))
    model.add(LSTM(units = 50, dropout = 0.2, recurrent_dropout = 0.2))
    model.add(Dense(1))
    model.compile(loss = 'mae',optimizer = 'adam')
    model.summary()
    return model
    
 
if __name__ == '__main__':
    #import data
    data = load_dataset()
    #encode col 4
    data = class_encode(data, 4)
    #create dataset
    dataset = convert_dataset(data,n_input = n_input)
    print(dataset)
    values = dataset.values.astype('float32')
    
    #seperate train dataset and validation dataset
    train = values[:n_train_hours,:]
    validation = values[-n_validation_hours:,:]
    x_train, y_train = train[:,:-1], train[:,-1]
    x_validation, y_validation = validation[:, :-1], validation[:, -1]
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_validation = x_validation.reshape(x_validation.shape[0], 1, 
                                        x_validation.shape[1])
    print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)
    
    #train model
    lstm_input_shape = (x_train.shape[1], x_train.shape[2])
    model = create_model(lstm_input_shape)
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
              validation_data = (x_validation, y_validation),verbose = 2)
    
    prediction = model.predict(x_validation)
    
    plt.plot(y_validation, color = 'blue', label = 'Actual')
    plt.plot(prediction, color = 'green', label = 'Prediction')
    plt.legend(loc = 'upper right')
    plt.show()
    