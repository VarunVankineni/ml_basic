# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:51:49 2018

@author: Varun
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib import style
from mnist import MNIST
style.use('ggplot')

a = 0.5   #learning Rate
maxitr = 1000  #Maximum number of iterations
res = 0.00001  #Convergence criterion


data = MNIST('E:/8thsem/ML/ass2')
imtrain, labtrain = data.load_training()
imtest, labtest = data.load_testing()
imtrain = np.array(imtrain)
imtest = np.array(imtest)
labtrain = np.array(labtrain)
labtest = np.array(labtest)
imtrain = np.c_[labtrain,imtrain]
imtest = np.c_[labtest,imtest]
imtrain = imtrain[imtrain[:,0]<=1]
imtest = imtest[imtest[:,0]<=1]



N1 = imtrain.shape[0]    #training samples of data
N2 = imtest.shape[0]    #testing samples of data
features = imtrain.shape[1] - 1  #number of features
classes = 2 #number of classes

ytrain = imtrain[:,0]
xtrain = imtrain[:,1:]

scaler = MinMaxScaler() #scaling using normalizer  
xtrain = scaler.fit_transform(xtrain)  
xtrain = np.c_[np.ones(N1),xtrain]
ytest = imtest[:,0]
xtest = imtest[:,1:]

xtest = scaler.transform(xtest)  
xtest = np.c_[np.ones(N2),xtest]
features = xtrain.shape[1]

    
w = np.array([0 for i in range(features)])  #initializing weights to 0

J = 0 #regression fit error
dw = 0 #intialize 
Jlist = []
Jlistt = []
expo = np.exp(-np.dot(xtrain,w))
Ji = np.sum(-(ytrain*np.log(1/(1+expo)))-((1-ytrain)*np.log(1-(1/(1+expo)))))/N1#calculating error after updating weights
Jlist.append(Ji)

expot = np.exp(-np.dot(xtest,w))
Jit = np.sum(-(ytest*np.log(1/(1+expot)))-((1-ytest)*np.log(1-(1/(1+expot)))))/N2#calculating error after updating weights
Jlistt.append(Jit)
donstop = True
for j in range(maxitr):  
    """
    if(donstop==True):
        
        for i in range(N1):#Stochastic Gradient Descent
            if(donstop==True):
               
                
                expo = np.exp(-np.dot(xtrain[i],w))
                dJ = ((1/(1+expo))-ytrain[i])*xtrain[i]/N1
                w = w - a*dJ #updating weights 
                
                
                #calculating training error after updating weights
                expo = np.exp(-np.dot(xtrain,w))
                Ji = np.sum(-(ytrain*np.log(1/(1+expo)))-((1-ytrain)*np.log(1-(1/(1+expo)))))/N1
                Jlist.append(Ji)
                
                #calculating testing error after updating weights
                expot = np.exp(-np.dot(xtest,w))
                Jit = np.sum(-(ytest*np.log(1/(1+expot)))-((1-ytest)*np.log(1-(1/(1+expot)))))/N2
                Jlistt.append(Jit)
                
                if(abs(J-Ji)<=res and i!=0 ): #checking for successive errors difference
                    donstop = False
                
                J = Ji #assigning error to be used in next iteration
                Jt = Jit
            
            else:
                break
            
    else:
        break
    """    
    
        
    
    #"""#Batch Gradient Design
    expo = np.exp(-np.dot(xtrain,w))
    dJ = np.sum(((1/(1+expo))-ytrain)*np.transpose(xtrain)/N1 , axis = 1)
    w = w - a*dJ #updating weights
    #calculating error after updating weights
    expo = np.exp(-np.dot(xtrain,w))
    Ji = np.sum(-(ytrain*np.log(1/(1+expo)))-((1-ytrain)*np.log(1-(1/(1+expo)))))/N1
    Jlist.append(Ji)
    #calculating error after updating weights
    expot = np.exp(-np.dot(xtest,w))
    Jit = np.sum(-(ytest*np.log(1/(1+expot)))-((1-ytest)*np.log(1-(1/(1+expot)))))/N2
    Jlistt.append(Jit)
    #"""
    if(abs(J-Ji)<=res and j!=0 ): #checking for successive errors difference
        J = Ji #assigning error to be used in next iteration
        Jt = Jit
        break
                
    J = Ji #assigning error to be used in next iteration
    Jt = Jit
   
#plot for cross entory error vs iterations    
trainerror, = plt.plot(Jlist,label = 'train error')
testerror, = plt.plot(Jlistt,label = 'test error')
plt.xlabel('No. of Iterations')
plt.ylabel('Error-J')
plt.legend(handles=[trainerror,testerror])
plt.show()

#accuracy of predictions for 
ydum = 1/(1+np.exp(-np.dot(xtrain,w)))
ydum[ydum>=0.5]=1
ydum[ydum<0.5]=0
Jfinal = ytrain-ydum
accuracy = 100 - 100*len(Jfinal[Jfinal!=0])/len(Jfinal)

ydum = 1/(1+np.exp(-np.dot(xtest,w)))
ydum[ydum>=0.5]=1
ydum[ydum<0.5]=0
Jfinalt = ytest-ydum
accuracyt = 100 - 100*len(Jfinalt[Jfinalt!=0])/len(Jfinalt)





















