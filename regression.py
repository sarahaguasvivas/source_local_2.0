#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import random

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h= tf.nn.sigmoid(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

if __name__== '__main__':
    train=0.8
    WINDOW_SIZE=125
    NUM_ADC=2
    data= pd.read_csv('alldata.csv').values

    ind= list(np.arange(data.shape[0]))

    indexes= random.sample(range(data.shape[0]), int(data.shape[0]*train))

    missing = list(set(ind)- set(indexes))

    training_data= data[indexes, data.shape[1]-2]
    testing_data = data[missing, data.shape[1]-2]
    training_labels= data[indexes, -2:]
    testing_labels= data[missing, -2:]
    print(training_data.shape)
    X= tf.placeholder("float", [None, ])

