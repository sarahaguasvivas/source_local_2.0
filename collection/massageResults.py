#!/usr/bin/env python3

import struct
import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

NUM_ADC= 2
filename = sys.argv[1]
datafile= open(sys.argv[1]+'.csv', 'r')
datainfo= str(datafile.readlines()[0])
datai= datainfo.split(",")
data= [float(i) for i in datai]
labelsi= filename.split("_")
labels = [float(i) for i in labelsi]

Data= np.array(data)

if len(Data) % 2 == 0:
    Data= np.reshape(Data, (-1, NUM_ADC))
else:
    Data= np.reshape(Data[:-1], (-1, NUM_ADC))

Data[:, 0] = np.abs(Data[:, 0] - labels[0])
Data[:, 1] = np.abs((Data[:, 1] - np.deg2rad(labels[1])))
Data[:, 1] = (Data[:, 1] + np.pi) % (2*np.pi) - np.pi

print(np.mean(Data, axis=0))
print(np.std(Data, axis=0))

for i in range(NUM_ADC):
    plt.plot(Data[:, i])
plt.show()
