#!/usr/bin/env python2

from __future__ import print_function
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential
from keras.losses import cosine_proximity, mean_squared_error
from sklearn import preprocessing
from keras.activations import exponential, linear
import pandas as pd
import numpy as np
from tensorflow.losses import huber_loss
import random
from keras import backend as K
import sys
import tensorflow as tf
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from scipy import signal
WAVELET_THRESHOLD=0.1
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()
def custom_loss(y_true, y_pred):
    #return huber_loss(y_true, y_pred)
    return 100*180*mean_squared_error(y_true, y_pred)
    #return cosine_proximity(y_true, y_pred)

def custom_activation(x):
    return K.sigmoid(x)-0.5
    #return K.tanh(x)
    #return exponential(x)
    #return 10*linear(x)

def model_function(data, labels, test, lab_test):
    print(lab_test)
    data= data.reshape(data.shape[0], data.shape[1], 1).astype(np.float)
    test= test.reshape(test.shape[0], test.shape[1], 1).astype(np.float)

    model= Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(250, 1)))
    model.add(MaxPooling1D(pool_size=5))

    model.add(Flatten())
    model.add(Dense(input_dim= 4000, output_dim=10, activation=custom_activation))
    model.add(Dense(input_dim= 25, output_dim=2, activation=custom_activation))
    model.add(Dropout(0.3))
    model.compile(loss=custom_loss, optimizer='adam')
    print(lab_test)
    print('here')
    history= model.fit(data, labels, batch_size=10, nb_epoch=500,  verbose=2, validation_data=(test, lab_test))
    predictions=model.predict(test, batch_size=1)
    print(predictions[:, 0]*180)
    print(lab_test[:, 0]*180)
    print(predictions[:, 1]*100)
    print(lab_test[:, 1]*100)
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
    training_data= np.empty((0, 250))
    testing_labels= np.empty((0, 2))
    testing_data= np.empty((0, 250))
    testing_labels1= data[missing, -2:]
    for i in range(training_data1.shape[0]):
        widths= np.arange(1, 31)
        cwtmatr= signal.cwt(training_data1[i, :], signal.ricker, widths)
        if np.max(np.abs(cwtmatr))>=WAVELET_THRESHOLD:
            training_labels= np.vstack((training_labels, training_labels1[i, :]))
            training_data= np.vstack((training_data, training_data1[i, :]))

    for i in range(testing_data1.shape[0]):
        widths= np.arange(1, 31)
        cwtmatr1= signal.cwt(testing_data1[i, :], signal.ricker, widths)
        if np.max(np.abs(cwtmatr1))>=WAVELET_THRESHOLD:
            testing_labels= np.vstack((testing_labels, testing_labels1[i, :]))
            testing_data= np.vstack((testing_data, testing_data1[i, :]))
    training_labels[:, 0]= training_labels[:, 0]/180
    testing_labels[:, 0]= testing_labels[:,0]/180
    training_labels[:, 1]= training_labels[:, 1]/100
    testing_labels[:, 1]= testing_labels[:, 1]/100
    model= model_function(training_data, training_labels, testing_data, testing_labels)
