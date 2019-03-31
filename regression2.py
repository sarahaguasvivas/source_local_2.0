import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflowas as tf

def neural_net_model(X, input_dim)
    W1= tf.Variable(tf.random_uniform([input_dim, 10]))
    b1= tf.Variable(tf.zeros([10]))
    layer1= tf.add(tf.matmul(X, W1), b1)
    layer2= tf.nn.relu(layer1)

    W2= tf.Variable(tf.random_uniform([10, 10]))
    b2= tf.Variable(tf.Variable(tf.zeros[10]))
    layer2= tf.add(tf.matmul(layer1, W2), b2)
    layer2= tf.nn.relu(layer2)

    W0= tf.Variable(tf.random_uniform([10, 1]))
    b0= tf.Variable(tf.zeros[1])
    output= tf.add(tf.matmul(layer2, W0), b0)

    return output


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver= tf.train.Saver()
    for i in range(100):
        for j in range(X_train)
