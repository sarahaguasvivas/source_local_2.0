#!/usr/bin/env python
import struct
import socket
import sys

IP_1= '192.168.1.3'

BUFFER_SIZE= 7000

filename= str(sys.argv[1])+'.csv'
sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP_1, 5005))
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
