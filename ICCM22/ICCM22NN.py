#!/usr/bin/env python2
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
from sklearn.model_selection import train_test_split

WINDOW_SIZE= 25
NUM_ADC= 2

sess = tf.Session()
K.set_session(sess)
#np.set_printoptions(threshold=np.nan)

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
    cos_th_hat= y_pred[:, 2]
    cos_th_true= y_true[:, 2]
    sin_th_hat= y_pred[:, 1]
    sin_th_true= y_true[:, 1]
    coseno= cos_th_hat*cos_th_true + sin_th_hat*sin_th_true
    return K.abs(r_true*r_true + r_hat*r_hat - 2.0*r_true*r_hat*coseno)

def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)

def custom_activation(x):
    return 10*K.tanh(x)

def model_theta(data, labels, test, lab_test):
    model= Sequential()
    #model.add(Conv1D(kernel_size=6, filters=4))
    #model.add(Conv1D(kernel_size=4, filters=12))
    #model.add(Conv1D(kernel_size=2, filters=12))
    model.add(Flatten())
    model.add(Dense(20, activation= 'relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dense(10, activation= 'relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dense(2, activation= None, kernel_regularizer=regularizers.l2(0.02)))

    rms= RMSprop(lr=1e-3)
    model.compile(loss='mse', optimizer='adam')
    history= model.fit(data, labels, batch_size=1000, nb_epoch=1000,  \
                    verbose=1, validation_data=(test, lab_test))

    predictions=model.predict(test, batch_size=1)

    np.savetxt('predictions_theta.csv', predictions)
    np.savetxt('labels_test_theta.csv', lab_test)

    model.save("../model_theta1.hdf5")
    error= predictions-lab_test
    plt.plot(error[:, 0], label=r'$r_{error}[cm]$')

    #sin_cos_ratio= np.arctan2(predictions[:, 1], predictions[:, 2]) - np.arctan2(lab_test[:, 1], lab_test[:, 2])
    #plt.plot(sin_cos_ratio, label=r'$\frac{sin \theta}{cos \theta}_{error}$')
    #plt.title('Errors in Predictions')
    #plt.legend()
    #plt.show()
    print("mean_error r: ", np.mean(error[:, 0]))
    print("std_dev r: ", np.std(error[:, 0]))
    #print("mean_error sin/cos: ", np.mean(sin_cos_ratio))
    #print("std_dev sin/cos: ", np.std(sin_cos_ratio))
    ii=0

    for l in model.layers:
        print(str(l.input_shape) + ' ' + str(l.output_shape))
        ii+=1

def model_r(data, labels, test, lab_test):
    model= Sequential()
    model.add(Flatten())
    #model.add(Dense(8, activation = custom_activation, kernel_initializer= 'normal',\
    #                                    kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dense(10, activation = custom_activation, kernel_initializer= 'normal', \
                                      kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dense(8, activation = custom_activation, kernel_initializer= 'normal', \
                                        kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dense(1, activation = None, kernel_initializer= 'normal', \
                                        kernel_regularizer=regularizers.l2(0.02)))

    rms= RMSprop(lr=1e-4)
    model.compile(loss='mae', optimizer=rms)
    history= model.fit(data, labels, batch_size=5, nb_epoch=5000,  \
                    verbose=1, validation_data=(test, lab_test))

    predictions=model.predict(test, batch_size=1)

    np.savetxt('predictions_r.csv', predictions)
    np.savetxt('labels_test_r.csv', lab_test)
    print(predictions)
    print(lab_test)
    model.save("../model_r.hdf5")
    error= predictions-lab_test
    plt.plot(error[:, 0], label=r'$r_{error}[cm]$')

    #sin_cos_ratio= np.arctan2(predictions[:, 1], predictions[:, 2]) - np.arctan2(lab_test[:, 1], lab_test[:, 2])
    #plt.plot(sin_cos_ratio, label=r'$\frac{sin \theta}{cos \theta}_{error}$')
    #plt.title('Errors in Predictions')
    #plt.legend()
    #plt.show()
    print("mean_error r: ", np.mean(error[:, 0]))
    print("std_dev r: ", np.std(error[:, 0]))
    #print("mean_error sin/cos: ", np.mean(sin_cos_ratio))
    #print("std_dev sin/cos: ", np.std(sin_cos_ratio))
    ii=0

    for l in model.layers:
        print(str(l.input_shape) + ' ' + str(l.output_shape))
        ii+=1

if __name__== '__main__':
    train = 0.7

    train_r = True
    train_theta= False

    data= pd.read_csv('../collection/data_ICCM_June_10.csv').values
    print(data)
    Data= data[:, :-2]
    labels = np.reshape(data[:, -1], (-1, 1))

    angles= np.reshape(np.deg2rad(data[:,-2]), (-1, 1))
    labels= np.concatenate((labels, np.sin(angles)), axis=1)
    labels= np.concatenate((labels, np.cos(angles)), axis=1)

    training_data, testing_data, training_labels, testing_labels = train_test_split(
                         Data, labels, test_size=1-train, random_state=42)

    training_data= np.reshape(training_data, (-1, WINDOW_SIZE*NUM_ADC, 1))
    testing_data = np.reshape(testing_data, (-1 ,WINDOW_SIZE*NUM_ADC, 1))
    if train_r:
        model= model_r(training_data, training_labels[:, 0], testing_data, testing_labels[:, 0])
    if train_theta:
        model_theta= model_theta(training_data, training_labels[:, 1:], testing_data, testing_labels[:, 1:])
    sess.close()
