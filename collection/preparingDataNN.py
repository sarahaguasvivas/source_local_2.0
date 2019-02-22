#!/usr/bin/env python3

import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

listFiles= os.listdir('data_Feb_4')
NUM_ADC= 2
WINDOW_SIZE= 25

Data= pd.DataFrame()

maximum= 0
for i in listFiles:
    filefile= open(os.path.join('data_Feb_4', i))
    data= str(filefile.readlines()[0])
    datai= data.split(",")
    data= [float(i)/268372897.0 for i in datai]
    if np.max(data) > maximum:
        maximum= np.max(data)
    widths= np.arange(1, 31)
    if len(data) % NUM_ADC != 0:
        data= data[:-1]
    data= np.reshape(data, (-1, NUM_ADC))
    cwtmatr= signal.cwt(data[:, 1], signal.ricker, widths)
    """
    ### For display:
    plt.subplot(211)
    plt.imshow(cwtmatr, extent=[0, len(data)//2, 1, 31],  cmap='seismic', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    i= i.replace('.csv', '')
    labelTitle= i.split("_")

    print(labelTitle)
    plt.title('Wavelet Decomposed Signals for ' +  str(r'$\theta$ = ') + str(labelTitle[0]) + "$\degree, r=$ " + str(labelTitle[1]) + str(" cm"))
    plt.xlabel('timestamp')
    plt.ylabel('Spectrum Widths')
    plt.colorbar(orientation='horizontal')
    plt.subplot(212)
    plt.plot(data[:, 0], 'gray', label= 'ADC1')
    plt.plot(data[:, 1], 'k', label='ADC2')
    plt.title('Raw Signal')
    plt.margins(x=0)
    plt.xlabel('timestamp')
    plt.ylabel('Normalized Signal')
    plt.legend()
    plt.show()
    # Rolling window:
    print(cwtmatr.shape)
    print(len(data))
    """
    i= i.replace('.csv', '')
    labelTitle= i.split("_")
    labelTitle= [float(i) for i in labelTitle]

    for ii in range(0, data.shape[0], WINDOW_SIZE):
        # ii is index starting window
        window_spectrum= cwtmatr[:, ii:ii+WINDOW_SIZE]
        window_data= data[ii:ii+WINDOW_SIZE, :]

        if (np.max(window_spectrum)>0.1):
            line= np.append(np.reshape(window_data, (1, -1)), np.reshape(np.array(labelTitle), (1, -1)), axis= 1)
            Data= Data.append(pd.DataFrame(line), ignore_index= True)
print("Datafile saved!")
Data.to_csv("allDataSmallerWindow.csv", index=False)

