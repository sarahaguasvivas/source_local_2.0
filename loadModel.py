#!/usr/bin/env python
import sys
import h5py
import numpy as np
from keras import backend as K
from keras.models import load_model
import keras.losses
#np.set_printoptions(threshold=np.nan)

def custom_loss(y_true, y_pred):
    r_hat = y_pred[:, 1]
    r_true = y_true[:, 1]
    th_hat= y_pred[:, 0]
    th_true= y_true[:, 0]
    coseno= K.cos(th_hat-th_true)
    return K.abs(r_true**2 + r_hat**2 - 2*r_true*r_hat*coseno)

keras.losses.custom_loss=custom_loss

model= load_model('configurationC.hdf5')
textFile= open("wr.txt", "w")

inp= model.input
outputs = [layer.output for layer in model.layers]          # all layer outputs
functor = K.function([inp, K.learning_phase()], outputs)   # evaluation function

layer= int(sys.argv[1])
print(model.summary())
input_size= model.layers[layer].input_shape
first_input= model.layers[0].input_shape
weights= model.layers[layer].get_weights()[0] #weights
biases= model.layers[layer].get_weights()[1] #biases
print("input_shape:")
print(input_size)
print("Shape Weights:")
print(weights.shape)
print("Shape biases:")
print(biases.shape)
print("layer config:")
print(model.layers[layer].get_config())

strWeights= str(list(weights))
strBiases= str(list(biases))

strWeights= strWeights.replace('[', '{')
strWeights= strWeights.replace(']', '}')
strWeights= strWeights.replace('dtype=float32),', '')
strWeights= strWeights.replace('array(', '')
strWeights= strWeights.replace(', dtype=float32)', '')
print()

textFile.write(strWeights)
strBiases= strBiases.replace('[', '{')
strBiases= strBiases.replace(']', '}')

print()
print(strBiases)

print(model.layers[0])
print("Keys: %s" % f.keys())
a_group_key= list(f.keys())[0]

test= np.ones((1, 10, 1))
layer_outs=functor([test, 4])
print(layer_outs)


data= list(f[a_group_key])
print(data)

