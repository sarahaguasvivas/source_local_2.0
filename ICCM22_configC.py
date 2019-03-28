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

WINDOW_SIZE= 5
NUM_ADC= 2

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
    r_hat = y_pred[:, 0]
    r_true = y_true[:, 0]
    cos_th_hat= y_pred[:, 1]
    cos_th_true= y_true[:, 1]
    sin_th_hat= y_pred[:, 2]
    sin_th_true= y_true[:, 2]
    coseno= cos_th_hat*cos_th_true + sin_th_hat*sin_th_true

    return K.abs(r_true**2 + r_hat**2 - 2*r_true*r_hat*coseno)

def custom_activation(x):
    return K.tanh(x)

def model_function(data, labels, test, lab_test):
    model= Sequential()
    model.add(Conv1D(filters=2, kernel_size=2))
    model.add(Conv1D(filters=2, kernel_size=2))
    model.add(Flatten())
    model.add(Dense(20, activation= None))
    model.add(Dense(3, activation=None))

    rms= RMSprop(lr=1e-5)
    model.compile(loss=custom_loss,optimizer=rms)
    history= model.fit(data, labels, batch_size=100, nb_epoch=700,  verbose=1, validation_data=(test, lab_test))
    predictions=model.predict(test, batch_size=1)
    model.save("configurationC.hdf5")
    print("predictions-ground_truth:")
    print("predictions shape:", predictions.shape)
    print("labels test shape: ", lab_test.shape)
    plt.plot(predictions-lab_test)
    plt.title('Errors in Predictions')
    plt.show()
    ii=0

    for l in model.layers:
        print(str(l.input_shape) + ' ' + str(l.output_shape))
        ii+=1
if __name__== '__main__':
    train=0.70
    data= pd.read_csv('collection/allDataSmallerWindow2.csv').values

    ind= list(np.arange(data.shape[0]))

    indexes= random.sample(range(data.shape[0]), int(data.shape[0]*train))

    missing = list(set(ind)- set(indexes))

    training_data= data[indexes, :data.shape[1]-2]
    testing_data = data[missing, :data.shape[1]-2]

    training_labels= np.reshape(data[indexes, 1], (-1, 1))
    testing_labels= np.reshape(data[missing, 1], (-1, 1))

    training_angles= np.reshape(data[indexes, 0], (-1, 1))
    testing_angles= np.reshape(data[missing, 0], (-1, 1))

    training_labels= np.concatenate((training_labels, np.cos(training_angles)), axis=1)
    training_labels= np.concatenate((training_labels, np.sin(training_angles)), axis=1)

    testing_labels= np.concatenate((testing_labels, np.cos(testing_angles)), axis=1)
    testing_labels= np.concatenate((testing_labels, np.sin(testing_angles)), axis=1)
    # labels: (r, cos(th), sen(th))

    training_data= training_data.reshape(-1, WINDOW_SIZE*NUM_ADC, 1)
    testing_data= testing_data.reshape(-1, WINDOW_SIZE*NUM_ADC, 1)

    print(training_data.shape, testing_data.shape)
    print(training_labels.shape, testing_labels.shape)
    model= model_function(training_data, training_labels, testing_data, testing_labels)
    sess.close()
