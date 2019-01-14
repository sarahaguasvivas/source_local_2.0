#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import struct
import sys
import pandas as pd

filename= str(sys.argv[1])+'.csv'
data= pd.read_csv(filename, sep=',')
print(data.shape)
plt.plot(range(data.shape[0]), data.iloc[:, 0])
plt.plot(range(data.shape[0]), data.iloc[:, 1])
plt.ylim(0, 1)
plt.show()
