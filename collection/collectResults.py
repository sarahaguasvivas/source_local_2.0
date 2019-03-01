#!/usr/bin/env python
import struct
import socket
import sys

IP_1= '192.168.50.201'
IP_2= '192.168.50.129'
IP_3= '192.168.50.173'
IP_4= '192.168.50.37'
BUFFER_SIZE= 400000

filename= str(sys.argv[1])+'.csv'
sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP_3, 5005))
print("Connection established!")
try:
    listl=[]
    while (1):
        data= sock.recv(BUFFER_SIZE)
        if len(str(data))>0:
            str1= str(len(data)/4) + "f"
            data= struct.unpack(str1, data)
            listl+= list(data)
            print(data)
except KeyboardInterrupt:
    print("saving data file")
    filef= open(filename, 'w')
    listl= ",".join(str(bit) for bit in listl)
    filef.write(listl)
