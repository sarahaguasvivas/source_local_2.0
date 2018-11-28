#!/usr/bin/env python3

import numpy as np
import os
import sys

path= 'params'

files=os.listdir(path)
for name in files:
    print(np.load('params/' + name))
