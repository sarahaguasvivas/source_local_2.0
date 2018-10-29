#!/usr/bin/env python
import struct
import socket
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
BUFFER_SIZE=2*4*500
NUM_ADC=2

IP_1= '192.168.50.201'
IP_2= '192.168.50.129'
IP_3= '192.168.50.173'
filename= open(str(sys.argv[1])+'.txt',"w")
sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP_3, 5005))
print("Connection established!")
try:
	while (1):
		data= sock.recv(BUFFER_SIZE)
		#str1= str(len(data)/4) + "f"
		#Window= struct.unpack(str1, data)
		filename.write(str(data))

except Exception as e:
	print(str(e))

