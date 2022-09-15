# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:30:51 2018

@author: Varun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib import style

#statics
a = 0.1 #learning rate
maxitr = 1000#maximum number of iterations
con = 0.000001#convergence criterion
variant = 'BGD'#SDG or BGD as required
#import data, assign classes, scale 
data = pd.read_csv('E:/8thsem/ML/ass3/30_train_features.csv')
test = pd.read_csv('E:/8thsem/ML/ass3/30_test_features.csv')
data = np.array(data)
test = np.array(test)

data[:,-1][data[:,-1]<=300]=0
data[:,-1][np.logical_and(data[:,-1]>300,data[:,-1]<450)]=1
data[:,-1][data[:,-1]>=450]=2

test[:,-1][test[:,-1]<=300]=0
test[:,-1][np.logical_and(test[:,-1]>300,test[:,-1]<450)]=1
test[:,-1][test[:,-1]>=450]=2

scaler = StandardScaler()
data[:,:-1] = scaler.fit_transform(data[:,:-1])
test[:,:-1] = scaler.transform(test[:,:-1])

features = data.shape[1]-1
samples = data.shape[0]
samples_t = test.shape[0]
classes = np.unique(data[:,-1]).shape[0]

onehoty = np.eye(classes)[data[:,-1].astype('int64')]
onehoty_t = np.eye(classes)[test[:,-1].astype('int64')]


batchsize = samples


#placeholders for input varaibles
x = tf.placeholder(tf.float32, [batchsize,features], name = "x")
y = tf.placeholder(tf.float32, [batchsize,classes], name = "y")

x_t = tf.placeholder(tf.float32, [samples_t,features], name = "x_t")
y_t = tf.placeholder(tf.float32, [samples_t,classes], name = "y_t")

#variables to optimize
w = tf.Variable(tf.random_normal(shape = [features,classes], stddev = 0.01) , name = "weights")
b = tf.Variable(tf.zeros([1,classes]), name = "bias")

#prediction equation 
logit = tf.matmul(x,w) + b
logit_t = tf.matmul(x_t,w) + b
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logit, labels = y)
entropy_t = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logit_t, labels = y_t)
#loss
loss = tf.reduce_mean(entropy,name= 'loss')
loss_t = tf.reduce_mean(entropy_t,name= 'loss_t')

#algorithm to use
algo = tf.train.GradientDescentOptimizer(learning_rate = a).minimize(loss)

#session initiate
sess = tf.Session()
#initialize variables with default values already given
sess.run(tf.global_variables_initializer())



#initialize the losses to plot mean cross entropy, BGD variant
losses = np.empty(shape=[1],dtype=float)
losses[0] = sess.run(loss,feed_dict={x:data[:,:-1].reshape(samples,features),y:onehoty.reshape(samples,classes)})
losses_t = np.empty(shape=[1],dtype=float)
losses_t[0] = sess.run(loss_t,feed_dict={x_t:test[:,:-1].reshape(samples_t,features),y_t:onehoty_t.reshape(samples_t,classes)})


#train to optimize variables, BGD variant
for i in range(maxitr):  
    sess.run(algo,feed_dict={x:data[:,:-1].reshape(batchsize,features),y:onehoty.reshape(batchsize,classes)})
    losses = np.append(losses,sess.run(loss,feed_dict={x:data[:,:-1].reshape(batchsize,features),y:onehoty.reshape(batchsize,classes)}))
    losses_t = np.append(losses_t,sess.run(loss_t,feed_dict={x_t:test[:,:-1].reshape(samples_t,features),y_t:onehoty_t.reshape(samples_t,classes)}))
    if(abs(losses[-1]-losses[-2])<con):
        break


#calculate final outcomes
finalw,finalb = sess.run([w,b])
total = np.exp(np.dot(data[:,:-1],finalw)+finalb)
total_t = np.exp(np.dot(test[:,:-1],finalw)+finalb)

#manipulate for prediction
for i in range(classes):
    dum = total[:,i]/np.sum(total,1)
    dum_t = total_t[:,i]/np.sum(total_t,1)
    if(i==0):
        y_predicted = dum
        y_predicted_t = dum_t
    else:
        y_predicted = np.c_[y_predicted,dum]
        y_predicted_t = np.c_[y_predicted_t,dum_t]

maxcol = np.max(y_predicted,1)
maxcol_t = np.max(y_predicted_t,1)

for i in range(samples):
    y_predicted[i,:][y_predicted[i,:]!=maxcol[i]]=0

    y_predicted[i,:][y_predicted[i,:]==maxcol[i]]=1
for i in range(samples_t):
    y_predicted_t[i,:][y_predicted_t[i,:]!=maxcol_t[i]]=0
    y_predicted_t[i,:][y_predicted_t[i,:]==maxcol_t[i]]=1

diff = np.sum((onehoty - y_predicted)**2,1)/(classes-1)
diff_t = np.sum((onehoty_t - y_predicted_t)**2,1)/(classes-1)
accuracy = diff[diff==0].shape[0]/diff.shape[0]
accuracy_t = diff_t[diff_t==0].shape[0]/diff_t.shape[0]

yfinal = np.dot(y_predicted,np.unique(data[:,-1]))
yfinal_t = np.dot(y_predicted_t,np.unique(test[:,-1]))

#correlation coefficients 
corr_coefs = np.corrcoef(yfinal,data[:,-1],rowvar = False)
corr_coefs_t = np.corrcoef(yfinal_t,test[:,-1],rowvar = False)

#sensitivity and specificity
sensitivity_t,specificity_t = [],[]


positives_t = test[:,-1][test[:,-1]==0].shape[0]
positives_t2 = test[:,-1][test[:,-1]==1].shape[0]
positives_t3 = test[:,-1][test[:,-1]==2].shape[0]

sensitivity_t.append(yfinal_t[:positives_t][yfinal_t[:positives_t]==0].shape[0]/positives_t)
specificity_t.append(yfinal_t[positives_t:][yfinal_t[positives_t:]!=0].shape[0]/(samples_t-positives_t))

sensitivity_t.append(yfinal_t[positives_t:positives_t+positives_t2][yfinal_t[positives_t:positives_t+positives_t2]==1].shape[0]/positives_t2)
specificity_t.append(np.concatenate((yfinal_t[:positives_t],yfinal_t[-positives_t3:]))[np.concatenate((yfinal_t[:positives_t],yfinal_t[-positives_t3:]))!=1].shape[0]/(samples_t-positives_t2))

sensitivity_t.append(yfinal_t[positives_t+positives_t2:][yfinal_t[positives_t+positives_t2:]==2].shape[0]/positives_t3)
specificity_t.append(yfinal_t[:-positives_t3][yfinal_t[:-positives_t3]!=2].shape[0]/(samples_t-positives_t3))

#plt.plot(losses)
#plt.plot(losses_t)
