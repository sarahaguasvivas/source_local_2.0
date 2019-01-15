#!/usr/bin/env python
import struct
import socket
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
BUFFER_SIZE= 200000
NUM_ADC=2

IP_1= '192.168.50.201'
IP_2= '192.168.50.129'
IP_3= '192.168.50.173'
filename= str(sys.argv[1])+'.csv'
filef= open(filename, 'w')
filef.close()
sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP_3, 5005))
print("Connection established!")
try:
	while (1):
		data= sock.recv(BUFFER_SIZE)
                if data:
                    str1= str(len(data)/4) + "f"
                    Window= struct.unpack(str1, data)
                    print(len(Window))

                    if len(Window) % NUM_ADC != 0:
                        Window= Window[:-3]
                    Window= pd.DataFrame(np.reshape(Window, (-1, 2)))
                    Window.to_csv(filename, mode= 'a',index=False, header=False)
                    print('.')
except Exception as e:
	print(str(e))

