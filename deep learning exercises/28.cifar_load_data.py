# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:00:12 2020

@author: YE677
"""

from keras.datasets import cifar10
from scipy.misc import toimage
from matplotlib import pyplot as plt

#import data
(X_train, y_train), (X_validation, y_validation) = cifar10.load_data()


for i in range(9):
        plt.subplot(331+i)
        plt.imshow(toimage(X_train[i]))
plt.show()