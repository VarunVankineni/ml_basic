# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:11:21 2018

@author: Varun
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import style
style.use('ggplot')

a = 0.3   #learning Rate
cdw = a*0.5 #momentum fraction to be carried
maxitr = 10000  #Maximum number of iterations
res = 0.00001  #Convergence criterion

N = 1000    #synthetic samples of data
features = 2  #number of features
classes = 2 #number of classes
boundary = 'e' #use l for line and e for ellipse
algo = 'SGD' #BGD/SGD as required

#n-dimensional hyperspace, elipse declaration
hs_dim = np.array([40,30]) 
hs_cen = np.array([100,200]) #also used as centre for ellipse
elipse = np.array([12,10])


#randomly (uniform) set points in the hyperspace
xU = hs_cen + (np.random.uniform(size=(N,features))-0.5)*hs_dim 
if(boundary == 'l'): #calculate classes for linear boundary
    y = np.sum((xU - hs_cen)*hs_dim,1)
if(boundary == 'e'): #calculate classes for eliptical boundary 
    y = ((xU[:,0]-hs_cen[0])/elipse[0])**2 + ((xU[:,1]-hs_cen[1])/elipse[1])**2 - 1 
y[y>=0]=1 
y[y<0]=0

#linear classifier line
if(boundary == 'l'): 
    xU_fit = hs_cen[-1] - np.dot(hs_dim[:-1]/hs_dim[-1],np.transpose(np.sort(xU[:,:-1],axis = 0)-hs_cen[:-1]))

#eliptical classifier boundary
if(boundary == 'e'): 
    xU_fit1 = hs_cen[-1] + elipse[1]*np.sqrt(1-((np.arange(-elipse[0],elipse[0]+1))/elipse[0])**2)
    xU_fit2 = hs_cen[-1] - elipse[1]*np.sqrt(1-((np.arange(-elipse[0],elipse[0]+1))/elipse[0])**2)
    xU_fit = np.hstack((xU_fit1,xU_fit2))

#add back class column to syntheic points data
xU = np.c_[xU,y]

"""#plot for eliptical boundary with synthetic data points
liner = np.arange(hs_cen[0]-elipse[0],hs_cen[0]+elipse[0]+1)
liner = np.hstack((liner,np.flip(liner,0)))
liner = np.c_[liner,xU_fit]
inner = plt.scatter(xU[xU[:,2]==0][:,0],xU[xU[:,2]==0][:,1],label = 'Class 0, Inner')
outer = plt.scatter(xU[xU[:,2]==1][:,0],xU[xU[:,2]==1][:,1],label = 'Class 1, Outer')
boundary, = plt.plot(liner[:,0],liner[:,1],label = 'Actual Classifier', color = 'black')
plt.legend(handles=[boundary,inner,outer])
plt.xlim((hs_cen[0]-0.75*hs_dim[0],hs_cen[0]+0.75*hs_dim[0]))
plt.ylim((hs_cen[1]-0.75*hs_dim[1],hs_cen[1]+0.75*hs_dim[1]))
plt.show()
"""

"""#plot for linear boundary with synthetic data points
liner = np.c_[np.sort(xU[:,0],axis=0),xU_fit]
liner = liner[liner[:,-1]<=(hs_cen[-1]+abs(hs_dim[-1]/2))]
liner = liner[liner[:,-1]>(hs_cen[-1]-abs(hs_dim[-1]/2))]
inner = plt.scatter(xU[xU[:,2]==0][:,0],xU[xU[:,2]==0][:,1],label = 'Class 0')
outer = plt.scatter(xU[xU[:,2]==1][:,0],xU[xU[:,2]==1][:,1],label = 'Class 1')
boundary, = plt.plot(liner[:,0],liner[:,1],label = 'Actual Classifier', color = 'black')
plt.legend(handles=[boundary,inner,outer])
plt.xlim((hs_cen[0]-0.75*hs_dim[0],hs_cen[0]+0.75*hs_dim[0]))
plt.ylim((hs_cen[1]-0.75*hs_dim[1],hs_cen[1]+0.75*hs_dim[1]))
plt.show()
"""


y =xU[:,-1]
x =xU[:,:-1]
scaler = StandardScaler() #scaling using standard scaler to N[0,1]  
x = scaler.fit_transform(x)
if(boundary == 'e'):  
    x = np.c_[x,x[:,0]*x[:,1],x[:,0]**2,x[:,1]**2]#polynomial features to be used for eliptical boundary
x = np.c_[np.ones(N),x]
features = x.shape[1]

    
w = np.array([0 for i in range(features)])  #initializing weights to 0

J = 0 #regression fit error
dw = 0 #intialize 
Jlist = []
#error for set weights
expo = np.exp(-np.dot(x,w))
Ji = np.sum(-(y*np.log(1/(1+expo)))-((1-y)*np.log(1-(1/(1+expo)))))/N#calculating error after updating weights
Jlist.append(Ji)

for j in range(maxitr):  
    if(algo == 'SGD'): 
        #Stochastic Gradient Descent
        for i in range(N):
            expo = np.exp(-np.dot(x[i],w))
            dJ = ((1/(1+expo))-y[i])*x[i]/N
            w = w - a*dJ #updating weights 
    
    if(algo == 'BGD'): 
        #Batch Gradient Descent
        expo = np.exp(-np.dot(x,w))
        dJ = np.sum(((1/(1+expo))-y)*np.transpose(x)/N , axis = 1)
        w = w - a*dJ #updating weights
        
    expo = np.exp(-np.dot(x,w))
    Ji = np.sum(-(y*np.log(1/(1+expo)))-((1-y)*np.log(1-(1/(1+expo)))))/N#calculating error after updating weights
    Jlist.append(Ji)
    
    if(abs(J-Ji)<=res and j!=0 ): #checking for successive errors difference
        break 
    J = Ji #assigning error to be used in next iteration

plt.plot(Jlist)
plt.xlabel('No. of Iterations')
plt.ylabel('Error-J')
ydum = 1/(1+np.exp(-np.dot(x,w)))
ydum[ydum>=0.5]=1
ydum[ydum<0.5]=0
xU = np.c_[xU,ydum]
Jfinal = y-ydum
accuracy = 100 - 100*len(Jfinal[Jfinal!=0])/len(Jfinal)
"""#final plot of linear boundary and estimated classes
liner = np.c_[np.sort(xU[:,0],axis=0),xU_fit]
liner = liner[liner[:,-1]<=(hs_cen[-1]+abs(hs_dim[-1]/2))]
liner = liner[liner[:,-1]>(hs_cen[-1]-abs(hs_dim[-1]/2))]
boundary, = plt.plot(liner[:,0],liner[:,1],label = 'Actual Classifier', color = 'black')

inner = plt.scatter(xU[xU[:,3]==0][:,0],xU[xU[:,3]==0][:,1],label = 'Class 0, Predicted Inner')
outer = plt.scatter(xU[xU[:,3]==1][:,0],xU[xU[:,3]==1][:,1],label = 'Class 1, Predicted Outer')
plt.legend(handles=[boundary,inner,outer])
plt.xlim((hs_cen[0]-0.75*hs_dim[0],hs_cen[0]+0.75*hs_dim[0]))
plt.ylim((hs_cen[1]-0.75*hs_dim[1],hs_cen[1]+0.75*hs_dim[1]))
plt.show()
"""

"""#final plot of eliptical boundary and estimated classes
liner = np.arange(hs_cen[0]-elipse[0],hs_cen[0]+elipse[0]+1)
liner = np.hstack((liner,np.flip(liner,0)))
liner = np.c_[liner,xU_fit]
boundary, = plt.plot(liner[:,0],liner[:,1],label = 'Actual Classifier', color = 'black')
inner = plt.scatter(xU[xU[:,3]==0][:,0],xU[xU[:,3]==0][:,1],label = 'Class 0, Predicted Inner')
outer = plt.scatter(xU[xU[:,3]==1][:,0],xU[xU[:,3]==1][:,1],label = 'Class 1, Predicted Outer')
plt.legend(handles=[boundary,inner,outer])
plt.xlim((hs_cen[0]-0.75*hs_dim[0],hs_cen[0]+0.75*hs_dim[0]))
plt.ylim((hs_cen[1]-0.75*hs_dim[1],hs_cen[1]+0.75*hs_dim[1]))
plt.show()
"""







