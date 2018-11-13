#!/usr/bin/env python
import os
import sys
import struct
import numpy as np

WINDOW_SIZE= 125
NUM_ADC= 2

list_of_files = os.listdir('data1')
BigData= np.empty([1, WINDOW_SIZE*NUM_ADC+2])

for i in range(len(list_of_files)):
    Data= ""
    with open(str(os.path.join('data1', list_of_files[i])), 'r') as f:
        data= f.read()
        str1= str(len(data)/4)+"f"
        Data= Data + str(struct.unpack(str1, data))
    data= Data
    list_data= data.split(')(')
    data= {}
    dataData= []
    for j in range(len(list_data)):
        list_data[j]= list_data[j].replace('(', '')
        list_data[j]= list_data[j].replace(')', '')
        list_data[j]= list_data[j].replace('\n', '')
        data[j]= list_data[j].split(',')
        if len(data[j])>0:
            data[j]= [float(j) for j in data[j]]
            dataData+= data[j]
    print(len(dataData[:len(dataData)-(len(dataData)%(NUM_ADC*WINDOW_SIZE))]))
    data= np.reshape(dataData[:len(dataData)-(len(dataData) % (NUM_ADC*WINDOW_SIZE))], (-1, NUM_ADC*WINDOW_SIZE))

    #print(len(dataData[0:-1-(len(dataData)%(NUM_ADC*WINDOW_SIZE+1))]))
    #print(data)
    string_split= list_of_files[i].split('_')
    degrees= string_split[0]
    distance= string_split[2].split('.')[0]

    labels= np.tile([float(degrees), float(distance)], (data.shape[0], 1))
    BigData= np.vstack((BigData, np.concatenate((data, labels), axis=1)))

np.savetxt('alldata.csv', BigData, delimiter=',')
