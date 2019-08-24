#!/usr/bin/env python3

import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

listFiles= os.listdir('data_August_3')
print(listFiles)
NUM_ADC= 2
WINDOW_SIZE=25

WAVELET_THRESHOLD= 1.0
COEFFICIENT_ATTENUATION = 0.221

Data= pd.DataFrame()

def wavelet_threshold_attenuation(r):
#    y = -7E-05x5 + 0.0031x4 - 0.0477x3 + 0.3338x2 - 1.0378x + 1.2945
    return -7e-05*r**5 + 0.00318*r**4 - 0.0477*r**3 + 0.3338*r**2 - 1.0378*r + 1.2945
 #   return 1.3*WAVELET_THRESHOLD / np.exp(COEFFICIENT_ATTENUATION*r)

numfiles= 0
for i in listFiles:
    numfiles+=1
    filefile= open(os.path.join('data_August_3', i))
    data= str(filefile.readlines()[0])
    datai= data.split(",")
    data= [float(ei) for ei in datai]
    widths= np.arange(1, 31)
    if len(data) % NUM_ADC*WINDOW_SIZE != 0:
        data= data[:-1]
    data= np.reshape(data, (-1, NUM_ADC*WINDOW_SIZE))
    print(data.shape)
    i= i.replace('.csv', '')
    labelTitle= i.split("_")
    labelTitle= labelTitle[:2]
    labelTitle= [float(iii) for iii in labelTitle]
    count_this= 0

    labels = np.tile(labelTitle, (data.shape[0] , 1))

    line= np.append(data,labels, axis= 1)
    Data= Data.append(pd.DataFrame(line), ignore_index= True)
    count_this+=line.shape[0]
    print(line)
    print("file " +  i + " " + str(numfiles) + "/"+ \
                str(len(listFiles))+ " samples: " + str(count_this))

print("Datafile saved!")
print(Data.shape)
Data.to_csv("data_ICCM_June_10.csv", index=False)

