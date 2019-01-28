#!/usr/bin/env python3

from keras.models import Sequential
from keras.models import load_weights
import numpy as np

model= Sequential()
model.load_weights('weightsR.h5py')
print(model.get_weights())
