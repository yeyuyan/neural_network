# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:40:20 2020

@author: YE677
"""

from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot as plt

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

if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset.head(5))
    
    #check the tendance of data
    groups = [0,1,2,3,5,6,7]
    plt.figure()
    i = 1
    for group in groups:
        plt.subplot(len(groups),1,i)
        plt.plot(dataset.values[:,group])
        plt.title(dataset.columns[group], y = 0.5, loc = 'right')
        i = i+1
    plt.show()