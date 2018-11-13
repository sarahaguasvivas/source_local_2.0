#!/usr/bin/env python3

from __future__ import print_function
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random
def model_function(data, labels, test, lab_test):
    model= Sequential()
    model.add(Dense(input_dim=250, output_dim=500))
    model.add(Activation('relu'))
    model.add(Dense(input_dim=500, output_dim=2))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.fit(data, labels, batch_size=1, nb_epoch=10,  verbose=2, validation_data=(test, lab_test))
    print(model.predict(test, batch_size=1))
    return model
if __name__== '__main__':
    train=0.8
    WINDOW_SIZE=125
    NUM_ADC=2
    data= pd.read_csv('alldata.csv').values

    ind= list(np.arange(data.shape[0]))

    indexes= random.sample(range(data.shape[0]), int(data.shape[0]*train))

    missing = list(set(ind)- set(indexes))

    training_data= data[indexes, data.shape[1]-2]
    print(training_data.shape[1])
    testing_data = data[missing, data.shape[1]-2]
    training_labels= data[indexes, -2:]
    testing_labels= data[missing, -2:]
    model= model_function(training_data, training_labels, testing_data, testing_labels)


