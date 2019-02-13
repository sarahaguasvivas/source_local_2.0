#!/usr/bin/env python
"""
    Parallel Collection code for Khoi
        by Sarah Aguasvivas Manzano
    This could be extended to multirhreading

"""


import struct
import socket
import sys
from multiprocessing import Pool, Manager, Value, Queue
# Pool --> pool of cores
# Manager--> scheduler capable of making locks
# Value--> I forgot
# Queue--> allows you to share data across processes


IP_1= '192.168.50.201'
IP_2= '192.168.50.129'
IP_3= '192.168.50.173'
IP_4= '192.168.50.37'

NUM_ESP= 11
BUFFER_SIZE= 1000000
STARTING_TCP_PORT=  5100

def collection_function(ready_to_read, IP, TCP_PORT, q, espID):
    # 'q' is optional. However, I added it because most likely you will need
    # to share some sort of data structure across processes
    # go to main to see how to declare it. In this callback
    # function you can read and write in q from any process
    # However, this is a double-edged sword because you
    # have to work so you have a good idea of who wrote what
    # and when.
    filename= str(sys.argv[1])+ '_' + str(espID)'.csv'
    sock= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((IP, TCP_PORT))
    print("Connection established for esp # " + str(espID))
    try:
        listl=[]
        while (ready_to_read):
            data= sock.recv(BUFFER_SIZE)
            str1= str(len(data)/4) + "I"
            data= struct.unpack(str1, data)
            listl+= list(data)

    except KeyboardInterrupt:
        print("saving data file for esp #"+ str(espID))
        filef= open(filename, 'w')
        listl= ",".join(str(bit) for bit in listl)
        filef.write(listl)
        sock.close()

if __name__ == "__main__":
    print("Trying to Connect to PZTs")
    manager= Manager() #declaring manager object
    q= Manager.dict() #the processes will be sharing a dictionary (optional)

    ESPIPlist={}
    for i in range(NUM_ESP):
    #This is optional, you delcare these if you are sharing a data structure
        q[i]= []
    # These were the IPs that I needed to connect
    # change these to the ones you need
    ESPIPlist[0]='192.168.50.129'
    ESPIPlist[1]='192.168.50.112'
    ESPIPlist[2]='192.168.50.45'
    ESPIPlist[3]='192.168.50.201'
    ESPIPlist[4]='192.168.50.173'
    ESPIPlist[5]='192.168.50.101'
    ESPIPlist[6]='192.168.50.131'
    ESPIPlist[7]='192.168.50.73'
    ESPIPlist[8]='192.168.50.193'
    ESPIPlist[9]='192.168.50.105'
    ESPIPlist[10]='192.168.50.36'
    ready_to_read= manager.Value('ready_to_read', False) # example of using semaphores across all processes
    ready_to_read.value= False
    # In this case I used 1 process per ESP.
    # You might want to do 1 multithreaded process
    # for a cluster of esps if you have more than
    # you have cores in the server PC
    pool= Pool(processes=NUM_ESP)
    results=[] #this is optional, you can just apply_async but this worked for me
    for count in range(NUM_ESP):
        # Here you haven't run yet the collection_function. You are just setting up the processes
        results.append(
        pool.apply_async(collection_function, args(ready_to_read, ESPIPlist[count], STARTING_TCP_PORT+count, q, count))
        )

    pool.close()
    ready_to_read.value=True
    for result in results:
        result.get()

    pool.join()
