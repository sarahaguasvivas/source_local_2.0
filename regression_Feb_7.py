#!/usr/bin/env python3
from __future__ import print_function
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.losses import mean_absolute_error, mean_squared_error
from keras import layers
from sklearn import preprocessing
from keras.activations import exponential, linear
import keras
import pandas as pd
from keras.optimizers import Adam, RMSprop
from keras import regularizers
import numpy as np
from tensorflow.losses import huber_loss
import random
from keras import backend as K
import sys
import tensorflow as tf
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from scipy import signal

WAVELET_THRESHOLD=0.2
sess = tf.Session()
K.set_session(sess)
np.set_printoptions(threshold=np.nan)

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
    return mean_absolute_error(y_true, y_pred)
    #a= y_true[:, 1]
    #b= y_pred[:, 1]
    #d= y_true[:, 0]
    #e= y_pred[:, 0]
    #delthet=d-e
    #cosdelt= K.cos(delthet*np.pi)

#    c =a*a + b*b - 2*a*b*cosdelt
#    res = K.mean(c)
#    return K.sqrt(res)

def custom_activation(x):
    #return K.sigmoid(x)-0.5
    return K.tanh(x)
    #return exponential(x)
    #return 10*linear(x)

def model_function(data, labels, test, lab_test):
    model= Sequential()
    model.add(Conv1D(filters=8, kernel_size=4, input_shape = (50, 1)))
    model.add(Conv1D(filters=8, kernel_size=4))
    model.add(Conv1D(filters=8, kernel_size=4))
    model.add(Flatten())
    model.add(Dense(50, activation= 'relu'))
    model.add(Dense(20, activation= 'relu'))
    model.add(Dense(10, activation= None, kernel_regularizer= regularizers.l2(0.01)))
    model.add(Dense(1, activation=None))

    rms= RMSprop(lr=1e-4, clipvalue= 0.5)
    model.compile(loss='mae',optimizer=rms)
    history= model.fit(data, labels, batch_size=10, nb_epoch=7000,  verbose=1, validation_data=(test, lab_test))
    predictions=model.predict(test, batch_size=1)
    model.save("nn_regression_att2.hdf5")
    print("predictions-ground_truth:")
    #print(predictions-lab_test)
    print("predictions shape:", predictions.shape)
    print("labels test shape: ", lab_test.shape)
    plt.plot(predictions-lab_test)
    plt.title('Errors in Predictions')
    plt.show()
    ii=0

    for l in model.layers:
        print(str(l.input_shape) + ' ' + str(l.output_shape))
        #print(l.get_weights())
        #l.get_weights()[0].tofile("params/weights"+str(ii)+".txt", sep=',', format="%.7e")
        #l.get_weights()[1].tofile("params/bias"+str(ii)+".txt", sep=',', format="%.7e")
        #filename=open("params1/weights"+str(ii)+".txt", "w")
        #filename.write(str(l.get_weights()))

        ii+=1
if __name__== '__main__':
    train=0.90
    data= pd.read_csv('collection/allDataSmallerWindow.csv').values

    ind= list(np.arange(data.shape[0]))

    indexes= random.sample(range(data.shape[0]), int(data.shape[0]*train))

    missing = list(set(ind)- set(indexes))

    training_data= data[indexes, :data.shape[1]-2]
    testing_data = data[missing, :data.shape[1]-2]
    training_labels= data[indexes, -2:]
    testing_labels= data[missing, -2:]
    training_data= training_data.reshape(-1, 50, 1)
    testing_data= testing_data.reshape(-1, 50, 1)
    print(training_data.shape, testing_data.shape)
    print(training_labels.shape, testing_labels.shape)

    model= model_function(training_data, training_labels[:, 1], testing_data, testing_labels[:, 1])
