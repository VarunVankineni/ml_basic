# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:58:50 2018

@author: Varun
"""


import numpy as np
import pandas as pd
import matplotlib as plt

N = 100
features = 6
weights = np.array([2,8,-6,1,3,18,-7]) 
x = np.c_[np.ones(N),np.random.rand(100,6)]
y =np.dot(x,weights)


d1 = np.random.normal(0,0.0001,(100,features+1))
d2 = np.random.normal(0,0.0001,N)

x+=d1
y+=d2

del d1,d2


w = [10,10,10,10,10,10,10]
a = 0.005
maxitr = 10000
res = 0.00000001

J = 0

for j in range(maxitr):  
    Ji = 0
    dJ = np.dot(np.transpose(x),(np.dot(x,w)  - y))
    Ji = sum((np.dot(x,w)  - y)*(np.dot(x,w)  - y))/(2*N)
    w = w -a*dJ
       
    if(abs(J-Ji)<=res and j!=0):
        break
    J = Ji
    