#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import struct
#df= pd.read_csv('testDataTest.csv')
#plt.plot(df.iloc[:, 0])
#plt.plot(df.iloc[:, 1])
#plt.plot(df.iloc[:, 2])
Data=""
with open('testDataTest.txt', 'r') as f:
	data= f.read()
	str1= str(len(data)/4) + "f"
	Data= Data+str(struct.unpack(str1, data))  
data= Data
list_data= data.split(')(')
data= {}
dataData= []

for i in range(len(list_data)):
	list_data[i]= list_data[i].replace('(','' )
	list_data[i]= list_data[i].replace(')', '')
	list_data[i]= list_data[i].replace('\n', '')
	data[i]= list_data[i].split(',')
	if len(data[i]) > 0:
		data[i]= [float(j) for j in data[i]]
		dataData+= data[i]

plt.scatter(range(len(dataData)), [float(i) for i in dataData])
plt.show()
data= np.reshape(dataData, (-1, 2))
print(data.shape)
plt.plot(range(data.shape[0]), data[:, 0])
plt.plot(range(data.shape[0]), data[:, 1])
plt.show()
"""
df= pd.read_csv('testData2.csv')
plt.plot(df.iloc[:, 0])
plt.plot(df.iloc[:, 1])
plt.plot(df.iloc[:, 2])

plt.show()

df= pd.read_csv('testData3.csv')
plt.plot(df.iloc[:, 0])
plt.plot(df.iloc[:, 1])
plt.plot(df.iloc[:, 2])

plt.show()

df= pd.read_csv('testData4.csv')
plt.plot(df.iloc[:, 0])
plt.plot(df.iloc[:, 1])
plt.plot(df.iloc[:, 2])

plt.show()
"""
