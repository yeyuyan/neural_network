# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:08:05 2020

@author: YE677
"""
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error 
import math

filename = 'international-airline-passengers.csv'

seed = 7
batch_size = 1
epochs = 100
footer = 0
look_back = 3

def create_dataset(dataset):
    dataX,dataY = [], []
    for i in range(len(dataset) - look_back):
        x = dataset[i:i+look_back, 0] #ddataset type: array of numpy
        dataX.append(x)
        y = dataset[i+look_back, 0]
        dataY.append(y)
        print('X: %s, Y: %s' % (x, y))
    return np.array(dataX),np.array(dataY)

def create_model():
    model = Sequential()
    model.add(LSTM(units = 4, input_shape = (1, look_back)))
    model.add(Dense(units=1))
    model.compile(loss = 'mean_squared_error',optimizer = 'adam')
    return model

if __name__ == '__main__':
    
    #set random seed
    np.random.seed(seed)
    #import data
    data = read_csv(filename, usecols = [1], engine = 'python', skipfooter = footer)
    dataset = data.values.astype('float32')
    
    #standarize data
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset)*0.67)
    validation_size = len(dataset)-train_size
    train,validation = dataset[0:train_size,:],dataset[train_size:len(dataset),:]
    
    X_train,y_train = create_dataset(train)
    X_validation,y_validation = create_dataset(validation)
    
    # transform data as [sample, time step, feature]
    X_train = np.reshape(X_train,(X_train.shape[0], 1 ,X_train.shape[1]))
    X_validation = np.reshape(X_validation,(X_validation.shape[0], 1 ,
                                            X_validation.shape[1]))
    
    #train model
    model = create_model()
    model.fit(X_train,y_train,epochs = epochs, batch_size = batch_size, 
              verbose = 2)

    #prediction
    predict_train = model.predict(X_train)
    predict_validation = model.predict(X_validation)
    
    #inverse standardization to ensure the veracity of MSE
    predict_train = scaler.inverse_transform(predict_train)
    y_train = scaler.inverse_transform([y_train])
    predict_validation = scaler.inverse_transform(predict_validation)
    y_validation = scaler.inverse_transform([y_validation])
    
    #evaluation
    train_score = math.sqrt(mean_squared_error(y_train[0],predict_train[:,0]))
    print('Train Score: %.2f RMSE' % train_score)
    validation_score = math.sqrt(mean_squared_error(y_validation[0],
                                                    predict_validation[:,0]))
    print('Validation Score: %.2f RMSE' % validation_score)    
    
    #show prediction in chart
    predict_train_plot = np.empty_like(dataset)
    predict_train_plot[:,:] = np.nan
    predict_train_plot[look_back:len(predict_train)+look_back,:] = predict_train
    
    predict_validation_plot = np.empty_like(dataset)
    predict_validation_plot[:,:] = np.nan
    predict_validation_plot[len(predict_train)+look_back*2 :len(dataset),
                            :] = predict_validation
    
    dataset = scaler.inverse_transform(dataset)
    plt.plot(dataset,color = 'blue')
    plt.plot(predict_train_plot,color = 'green')
    plt.plot(predict_validation_plot,color = 'red')
    plt.show()
    