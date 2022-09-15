# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:46:18 2018

@author: Varun
"""


import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LinearRegression



data = pd.read_csv('E:\\8thsem\\ML\\assignment1.csv',index_col=0)
data.dropna(inplace=True)
x= np.array(data.iloc[:,1:])
y= np.array(data.iloc[:,0])
features = x.shape[1]
N = x.shape[0]
for i in range(features):
    x[:,i] = 2*(x[:,i] - np.min(x[:,i]))/(np.max(x[:,i]) - np.min(x[:,i])) -1
    

reg = GradientBoostingClassifier(n_estimators  = 40, max_depth = 3)
reg.fit(X_train, y_train)
result.append(reg.score(X_test, y_test))
prediction1  =  pd.DataFrame(reg.predict(X_data))
prediction1.index  = data1[1].index



