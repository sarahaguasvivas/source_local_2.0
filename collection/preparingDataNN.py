#!/usr/bin/env python3

import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

listFiles= os.listdir('data_June_27')
print(listFiles)
NUM_ADC= 2
WINDOW_SIZE=100
LIMIT_NONE= 100
WAVELET_THRESHOLD= 0.20
COEFFICIENT_ATTENUATION = 2.0
Data= pd.DataFrame()

def wavelet_threshold_attenuation(r):
    return WAVELET_THRESHOLD / np.abs(1-COEFFICIENT_ATTENUATION*r)

numfiles= 0
for i in listFiles:
    numfiles+=1
    filefile= open(os.path.join('data_June_27', i))
    data= str(filefile.readlines()[0])
    datai= data.split(",")
    data= [float(ei) for ei in datai]
    widths= np.arange(1, 31)
    if len(data) % NUM_ADC != 0:
        data= data[:-1]
    data= np.reshape(data, (-1, NUM_ADC))
    cwtmatr= signal.cwt(data[:, 1], signal.ricker, widths)

    # SEPARATING r AND theta FROM STRING:
    i= i.replace('.csv', '')
    labelTitle= i.split("_")
    labelTitle= labelTitle[:2]
    labelTitle= [float(iii) for iii in labelTitle]
    count_this= 0

    # PERFORMING WAVELET DECOMPOSITION TO DETECT IMPACT EVENTS:
    for ii in range(0, data.shape[0], WINDOW_SIZE):
        window_spectrum= cwtmatr[:, ii:ii+WINDOW_SIZE]
        window_data= data[ii:ii+WINDOW_SIZE, :]

        # WAVELET THRESHOLD DEPENDS ON DISTANCE TO THE CENTER:
        cutoff = wavelet_threshold_attenuation(labelTitle[1]) #distance
        if (np.max(np.max(window_spectrum))>cutoff):
            line= np.append(np.reshape(window_data, (1, -1)), np.reshape(np.array(labelTitle), (1, -1)), axis= 1)
            Data= Data.append(pd.DataFrame(line), ignore_index= True)
            count_this+=1

    print("file " +  i + " " + str(numfiles) + "/"+ str(len(listFiles))+ " samples: " + str(count_this))

print("Datafile saved!")
print(Data.shape)
Data.to_csv("data_ICCM_June_10.csv", index=False)

