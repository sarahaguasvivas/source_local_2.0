#!/usr/bin/env python3

import struct
import sys
import csv
import matplotlib.pyplot as plt

NUM_ADC= 2
datafile= open(sys.argv[1]+'.csv', 'r')
datainfo= str(datafile.readlines()[0])
datai= datainfo.split(",")
data= [float(i) for i in datai]
Data= {}

for i in range(NUM_ADC):
    Data[i]= []


for i in range(len(data)-1):
    if int(data[i])==-1:
        Data[0].append(data[i+1])
    if int(data[i])==-2:
        Data[1].append(data[i+1])

plt.plot(Data[0])
plt.plot(Data[1])
plt.show()
print(len(Data[0]))
print(len(Data[1]))
