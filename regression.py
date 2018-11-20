#!/usr/bin/env python3

from __future__ import print_function
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random
from keras import backend as K
import sys
import tensorflow as tf
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from scipy import signal
WAVELET_THRESHOLD=0.1
def custom_activation(x):
    return K.tanh(x)

def model_function(data, labels, test, lab_test):
    model= Sequential()
    model.add(Dense(input_dim=250*30, output_dim=500))
    model.add(Activation(custom_activation))
    model.add(Dense(input_dim=500, output_dim=300))
    model.add(Activation(custom_activation))
    model.add(Dense(input_dim=300, output_dim= 1000))
    model.add(Activation(custom_activation))
    model.add(Dense(input_dim=1000, output_dim= 2))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(data, labels, batch_size=1, nb_epoch=100,  verbose=2, validation_data=(test, lab_test))
    print(model.predict(test, batch_size=1))
    print("Labels:")
    print(lab_test)
    return model

if __name__== '__main__':
    train=0.8
    WINDOW_SIZE=125
    NUM_ADC=2
    data= pd.read_csv('alldata.csv').values

    ind= list(np.arange(data.shape[0]))

    indexes= random.sample(range(data.shape[0]), int(data.shape[0]*train))

    missing = list(set(ind)- set(indexes))

    training_data1= data[indexes, :data.shape[1]-2]

    testing_data1 = data[missing, :data.shape[1]-2]
    training_labels1= data[indexes, -2:]
    training_labels= np.empty((0, 2))
    training_data= np.empty((0, 250*30))
    testing_labels= np.empty((0, 2))
    testing_data= np.empty((0, 250*30))
    testing_labels1= data[missing, -2:]

    for i in range(training_data1.shape[0]):
        widths= np.arange(1, 31)
        cwtmatr= signal.cwt(training_data1[i, :], signal.ricker, widths)
        if np.max(np.abs(cwtmatr))>=WAVELET_THRESHOLD:
            training_labels= np.vstack((training_labels, training_labels1[i, :]))
            training_data= np.vstack((training_data, cwtmatr.reshape(1, -1)))

    for i in range(testing_data1.shape[0]):
        widths= np.arange(1, 31)
        cwtmatr1= signal.cwt(testing_data1[i, :], signal.ricker, widths)
        if np.max(np.abs(cwtmatr1))>=WAVELET_THRESHOLD:
            testing_labels= np.vstack((testing_labels, testing_labels1[i, :]))
            testing_data= np.vstack((testing_data, cwtmatr1.reshape(1, -1)))

    model= model_function(training_data, training_labels, testing_data, testing_labels)
    print(testing_labels.shape)
    print(testing_data.shape)
