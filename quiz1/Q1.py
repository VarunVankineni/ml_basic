# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:07:30 2018

@author: Varun

Q1 :  clustering
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import DBSCAN,AgglomerativeClustering,KMeans
#from sklearn.mixture import BayesianGaussianMixture,GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from matplotlib import style
from matplotlib.patches import Ellipse
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
style.use('ggplot')

"""
Importing the data and standard scaling it 
"""
data = pd.read_csv('oldfaithful.csv')
scaler = StandardScaler()
data.iloc[:,:] = scaler.fit_transform(data.iloc[:,:])

"""
plot for intial observations
"""
#plt.xlabel(data.columns[0])
#plt.ylabel(data.columns[1]+' time')
#plt.scatter(data.iloc[:,0],data.iloc[:,1])

"""
DBSCAN clustering algorithm implementation
"""
clusterer = DBSCAN(eps = 0.4,min_samples = 10)#use eps = 0.52 for no outliers
db = clusterer.fit(data)
labels = db.labels_ #labels for the data points we fit the algorithm to 

"""
plot 
"""
#plt.xlabel(data.columns[0])
#plt.ylabel(data.columns[1]+' time')
#scatterplot = plt.scatter(data.iloc[:,0],data.iloc[:,1], c = labels)
"""
Gaussian Naive bayes classifier - as data is continuous
"""
Xtrain,Xtest,ytrain,ytest = train_test_split(data,labels,test_size = 0.3)
classifier = GaussianNB()
classifier.fit(Xtrain,ytrain)
ypred = classifier.predict(Xtest)
accuracy = classifier.score(Xtest,ytest)

"""
plots for comparing true labels to predicted labels
"""
#plt.xlabel(Xtest.columns[0])
#plt.ylabel(Xtest.columns[1]+' time')
#scatterplot = plt.scatter(Xtest.iloc[:,0],Xtest.iloc[:,1], c = ypred)#prediction data
#scatterplot = plt.scatter(Xtest.iloc[:,0],Xtest.iloc[:,1], c = ytest) #true data
"""
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1]+' time')
scatterplot = plt.scatter(data.iloc[:,0],data.iloc[:,1], c = labels)
plt.show()
"""

"""
Bivariate gaussian estimator
"""
class0 = data[labels==0]
class1 = data[labels==1]
mean0,mean1 = np.mean(class0),np.mean(class1)
std0,std1 = np.std(class0),np.std(class1)
rho0 = np.corrcoef(class0,rowvar = False)[0,1]
rho1 = np.corrcoef(class1,rowvar = False)[0,1]
X0,X1,Y0,Y1 = np.linspace(mean0[0]-2,mean0[0]+2,50),np.linspace(mean1[0]-2,mean1[0]+2,50),np.linspace(mean0[1]-2,mean0[1]+2,50),np.linspace(mean1[1]-2,mean1[1]+2,50)
X0,Y0 = np.meshgrid(X0,Y0)
X1,Y1 = np.meshgrid(X1,Y1)
Z1 = (((X1 - mean1[0])**2)/std1[0]) + (((Y1 - mean1[1])**2)/std1[1]) - ((2*rho1*(X1-mean1[0])*(Y1-mean1[1]))/(std1[0]*std1[1]))
P1 = (1/(2*np.pi*std1[0]*std1[1]*(np.sqrt(1-rho1**2))))*np.exp(-Z1/(2*(1-rho1**2)))

Z0 = (((X0 - mean0[0])**2)/std0[0]) + (((Y0 - mean0[1])**2)/std0[1]) - ((2*rho0*(X0-mean0[0])*(Y0-mean0[1]))/(std0[0]*std0[1]))
P0 = (1/(2*np.pi*std0[0]*std0[1]*(np.sqrt(1-rho0**2))))*np.exp(-Z0/(2*(1-rho0**2)))
"""
plot for class0
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X0,Y0,P0)
contour = ax.contourf(X0,Y0,P0,offset=-10)
scatter = ax.scatter(class0.iloc[:,0],class0.iloc[:,1],-10,c = 'skyblue')
ax.set_zlim(-10, 5)
ax.set_zlabel('probability density')
ax.set_xlabel(Xtest.columns[0])
ax.set_ylabel(Xtest.columns[1]+'time')
plt.show()

"""
plot for class1
"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1,Y1,P1)
contour = ax.contourf(X1,Y1,P1,offset=-10)
scatter = ax.scatter(class1.iloc[:,0],class1.iloc[:,1],-10,c = 'yellow')
ax.set_zlim(-10, 5)
ax.set_zlabel('probability density')
ax.set_xlabel(Xtest.columns[0])
ax.set_ylabel(Xtest.columns[1]+'time')
plt.show()

