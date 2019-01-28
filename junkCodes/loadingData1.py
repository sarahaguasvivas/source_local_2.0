import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

deg40p1= pd.read_csv('junkCodes/40deg1.txt', sep=" ", header= None)
deg40p1 = deg40p1.replace({r'\\r\\n': '', r'b': '', r"'": '', r'\\x': ''}, regex=True)

deg40p5= pd.read_csv('junkCodes/40deg5.txt', sep=" ", header= None)
deg40p5 = deg40p5.replace({r'\\r\\n': '', r'b': '', r"'": '', r'\\x': '', r'ff': ''}, regex=True)

deg40p10= pd.read_csv('junkCodes/40deg10.txt', sep=" ", header= None)
deg40p10 = deg40p10.replace({r'\\r\\n': '', r'b': '', r"'": '', r'\\x': '', r'ff': ''}, regex=True)

deg40p20= pd.read_csv('junkCodes/40deg20.txt', sep=" ", header= None)
deg40p20 = deg40p20.replace({r'\\r\\n': '', r'b': '', r"'": '', r'\\x': '', r'ff': '', r'fe': ''}, regex=True)

deg90p15= pd.read_csv('junkCodes/90deg15.txt', sep=" ", header= None)
deg90p15 = deg90p15.replace({r'\\r\\n': '', r'b': '', r"'": '', r'\\x': '', r'ff': '', r'fe': ''}, regex=True)

deg220p10= pd.read_csv('junkCodes/220deg10.txt', sep=" ", header= None)
deg220p10 = deg220p10.replace({r'\\r\\n': '', r'b': '', r"'": '', r'\\x': '', r'ff': '', r'fa': ''
                             , r'e': ''}, regex=True)

deg320p25= pd.read_csv('junkCodes/320deg25.txt', sep=" ", header= None)
deg320p25 = deg320p25.replace({r'\\r\\n': '', r'b': '', r"'": '', r'\\x': '', r'ff': '', r'fa': ''
                             , r'e': '', r'f': ''}, regex=True)

deg150p20= pd.read_csv('junkCodes/150deg20.txt', sep=" ", header= None)
deg150p20 = deg150p20.replace({r'\\r\\n': '', r'b': '', r"'": '', r'\\x': '', r'ff': '', r'fa': ''
                             , r'e': '', r'f': ''}, regex=True)


deg40p1= deg40p1.values
deg40p10= deg40p10.values
deg40p5= deg40p5.values
deg40p20= deg40p20.values
deg90p15= deg90p15.values
deg220p10= deg220p10.values
deg320p25= deg320p25.values
deg150p20 = deg150p20.values

deg40point1= []
deg40point10 = []
deg40point5 = []
deg40point20 = []
deg90point15 = []
deg220point10 = []
deg320point25= []
deg150point20= []

for i in range(deg40p1.shape[1]):
        if deg40p1[0][i]!="":
            deg40point1 += [int(deg40p1[0][i])]

for i in range(deg40p10.shape[1]):
        if deg40p10[0][i]!="":
            deg40point10 += [int(deg40p10[0][i])]
            
for i in range(deg40p5.shape[1]):
        if deg40p5[0][i]!="":
            deg40point5 += [int(deg40p5[0][i])]

for i in range(deg40p20.shape[1]):
        if deg40p20[0][i]!="":
            deg40point20 += [int(deg40p20[0][i])]

for i in range(deg90p15.shape[1]):
        if deg90p15[0][i]!="":
            deg90point15 += [int(deg90p15[0][i])]
            
for i in range(deg220p10.shape[1]):
        if deg220p10[0][i]!="":
            deg220point10 += [int(deg220p10[0][i])]
            
for i in range(deg320p25.shape[1]):
        if deg320p25[0][i]!="":
            deg320point25 += [int(deg320p25[0][i])]
            
for i in range(deg150p20.shape[1]):
        if deg150p20[0][i]!="":
            deg150point20 += [int(deg150p20[0][i])]
            
deg40point1= np.reshape(deg40point1, (-1,2))
deg40point10= np.reshape(deg40point10, (-1,2))
deg40point5= np.reshape(deg40point5, (-1,2))
deg40point20= np.reshape(deg40point20, (-1,2))
deg90point15= np.reshape(deg90point15, (-1,2))
deg220point10= np.reshape(deg220point10, (-1,2))
deg320point25= np.reshape(deg320point25, (-1,2))
deg150point20= np.reshape(deg150point20, (-1,2))
print("Data Loaded")


