#!/usr/bin/env python3

import struct
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

NUM_ADC= 4
datafile= open(sys.argv[1]+'.csv', 'r')
datainfo= str(datafile.readlines()[0])
datai= datainfo.split(",")
data= [float(i)/268372897.0 for i in datai]
#268372897.0
print(max(data))

Data= np.array(data)
print(Data.shape)

if len(Data) % 2 == 0:
    Data= np.reshape(Data, (-1, 2))
else:
    Data= np.reshape(Data[:-1], (-1, 2))


plt.plot(Data[:, 0])
plt.plot(Data[:, 1])
plt.show()
