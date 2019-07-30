#!/usr/bin/env python2
import struct
import socket
import sys

IP_1= '192.168.50.201'
IP_2= '192.168.50.129'
IP_3= '192.168.50.174'
IP_4= '192.168.50.37'
IP_5= '192.168.50.131'
IP_6= '10.0.0.7'

BUFFER_SIZE= 1000000

filename= str(sys.argv[1])+'.csv'
sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP_6, 5005))

print("Connection established!")
try:
    listl=[]
    while (1):
        data= sock.recv(BUFFER_SIZE)
        str1= str(len(data)/4) + "f"
        data= struct.unpack(str1, data)
        listl+= list(data)

except KeyboardInterrupt:
    print("saving data file")
    filef= open(filename, 'w')
    listl= ",".join(str(bit) for bit in listl)
    filef.write(listl)
