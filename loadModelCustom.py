#!/usr/bin/env python
import h5py
import numpy as np


f= h5py.File('nn_regression_att2.hdf5', 'r')

print(f['model_weights'])


