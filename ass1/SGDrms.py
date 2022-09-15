# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 00:09:33 2018

@author: Varun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


a = 0.1   #learning Rate
maxitr = 10000  #Maximum number of iterations
cfv = 0.9 #fraction to sum over
res = 0.001  #Convergence criterion

"""
N = 100    #synthetic samples of data
features = 6  #number of features
weights = np.array([2,8,-6,1,3,18,-7])   #pre determined weights
x = np.c_[np.ones(N),np.random.rand(100,6)]   #random assignment of data in x
y =np.dot(x,weights) #calculated y using weights


x += np.random.normal(0,0.0001,(100,features+1))  #random error incorporated into x
y += np.random.normal(0,0.0001,N) #random error incorporated into y


"""

#"""


data = pd.read_csv('E:\\8thsem\\ML\\assignment1.csv',index_col=0)  #imported data set
data.dropna(inplace=True) #dropping samples with unavailable data
x= np.array(data.iloc[:,1:]) #assigning features
y= np.array(data.iloc[:,0])  #assigning output

features,N = x.shape[1],x.shape[0]     

for i in range(features): #scaling using min_max scaler to [-1,1]
    x[:,i] = 2*(x[:,i] - np.min(x[:,i]))/(np.max(x[:,i]) - np.min(x[:,i])) -1
   
x = np.c_[np.ones(N),x] #adding x=1 to account for the constant term of the regression fit in matrix calculations

#"""


w = [0 for i in range(features+1)]  #initializing weights to 0
J = 0 #regression fit error
Jlist = []
v=0 #initialize
for j in range(maxitr):  
    Ji = 0 #error in the ith iteration/ this iteration
    for i in range(N):#Stochastic Gradient Descent-RMS prop
        dJ = np.dot(np.transpose(x[i,:]),(np.dot(x[i],w)  - y[i]))/N #grad J
        v = cfv*v + (1-cfv)*((dJ)**2) #calculation of the correction term for learning rate
        w = w -a*(v**(-0.5))*dJ #updating weights 
    Ji = sum((np.dot(x,w)  - y)*(np.dot(x,w)  - y))/(2*N) #calculating error after updating weights
    Jlist.append(Ji)
    if(abs(J-Ji)<=res and j!=0): #checking for successive errors difference
        break 
    J = Ji #assigning error to be used in next iteration


plt.plot(Jlist)
plt.xlabel('No. of Iterations')
plt.ylabel('Error-J')