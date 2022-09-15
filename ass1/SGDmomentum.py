

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 23:57:13 2018

@author: Varun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import style
style.use('ggplot')
a = 0.1   #learning Rate
cdw = a*0.5 #momentum fraction to be carried
maxitr = 10000  #Maximum number of iterations
res = 0.01  #Convergence criterion
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
data.fillna(data.median(),inplace=True) #imputing unavailable data with mean
x= np.array(data.iloc[:,1:]) #assigning features
y= np.array(data.iloc[:,0])  #assigning output

itr = 25
tw = 0
for k in range(itr):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  
    
    features,N = x_train.shape[1],x_train.shape[0]
    features2,N2 = x_test.shape[1],x_test.shape[0]
    
    scaler = StandardScaler() #scaling using standard scaler to N[0,1]  
    x_train = scaler.fit_transform(x_train)  
    x_test = scaler.transform(x_test)
    
    
    x_train = np.c_[np.ones(N),x_train]#adding x=1 to account for the constant term of the regression fit in matrix calculations
    x_test = np.c_[np.ones(N2),x_test]#adding x=1 to account for the constant term of the regression fit in matrix calculations
    
    
    #"""
    
    
    w = [0 for i in range(features+1)]  #initializing weights to 0
    J = 0 #regression fit error
    dw = 0 #intialize 
    Jlist = []
    for j in range(maxitr):  
        Ji = 0 #error in the ith iteration/ this iteration
        for i in range(N):#Stochastic Gradient Descent- momentum
            dJ = np.dot(np.transpose(x_train[i,:]),(np.dot(x_train[i],w)  - y_train[i]))/N #grad J
            dw = cdw*dw -a*dJ
            w = w + dw #updating weights 
        
        Ji = sum((np.dot(x_test,w)  - y_test)**2)/(N2) #calculating error after updating weights
        Jlist.append(Ji)
        
        if(abs(J-Ji)<=res and j!=0 ): #checking for successive errors difference
            break 
        J = Ji #assigning error to be used in next iteration
    tw+=w
    """
    plt.plot(Jlist)
    plt.xlabel('No. of Iterations')
    plt.ylabel('Error-J')
    """
tw = tw/itr

Jtotal = sum((np.dot(x_test,tw)  - y_test)**2)/(N2)


x = scaler.transform(x)
x = np.c_[np.ones(N+N2),x]
ycheck =np.dot(x,w)


"""

var = 5
slope, intercept = np.polyfit(x[:,var],ycheck,1)
abline_values = [slope * i + intercept for i in x[:,var]]
plt.plot(x[:,var], abline_values, 'b')
plt.scatter(x[:,var],y)
plt.title('Plot of estimated values vs actual values for Var'+str(var))

"""





















