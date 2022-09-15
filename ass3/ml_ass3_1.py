# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:32:33 2018

@author: AAQUIB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

raw_train = pd.read_csv('E:/8thsem/ML/ass3/30_train_features.csv')
raw_test = pd.read_csv('E:/8thsem/ML/ass3/30_test_features.csv')

features = np.array(list(raw_train.columns))
raw_label = np.array(raw_train[features[-1]])

label_1 = 1*(raw_label>300)
label_2 = []
for i in range(len(raw_label)):
    if (raw_label[i] < 300):
        label_2.append(0)
    elif (raw_label[i] < 450):
        label_2.append(1)
    else:
        label_2.append(2)

label_2 = np.array(label_2)

data = raw_train.iloc[:,0:30]
data['ylabel'] = label_1
data = data.sample(frac = 1)
#data.to_csv('E:/sem8/ml/Ass3/assignment3_train1.csv')

data = raw_train.iloc[:,0:30]
data['ylabel'] = label_2
data = data.sample(frac = 1)
#data.to_csv('E:/sem8/ml/Ass3/assignment3_train2.csv')


#df = pd.read_csv('E:/sem8/ml/Ass3/assignment3_train1.csv')

xtrain = np.array(data.iloc[:,0:30])
ytrain = np.array(data.iloc[:,30:31])

xtrain = np.array(data.iloc[:,0:30])
ytrain = np.array(data.iloc[:,30:31])

#-------------------------------------------------------------------------------------

test_data = pd.read_csv('E:/8thsem/ML/ass3/30_test_features.csv')

xtest = np.array(test_data.iloc[:,0:30])

raw_test_label = np.array(test_data.iloc[:,30:31])

ytest_1 = 1*(raw_test_label > 300)

ytest_2 = []

for i in range(len(raw_test_label)):
    if (raw_test_label[i] < 300):
        label_2.append(0)
    elif (raw_test_label[i] < 450):
        label_2.append(1)
    else:
        label_2.append(2)

ytest_2 = np.array(ytest_2)




#-------------------------------------------------------------------------------------

#tensorflow model

n_dim = xtrain.shape[1]
alpha = 0.01
epoch = 100
cost_history = np.empty(shape=[1],dtype=float)


X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.ones([n_dim,1]))

init = tf.initialize_all_variables()
init = tf.global_variables_initializer


yhat = 1/(1+ tf.exp(-1*tf.matmul(X, W)))
cost = tf.reduce_mean(Y*tf.log(yhat) + (1-Y)*tf.log(1-yhat))
training_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)


sess = tf.Session()
sess.run(init)

for epoch in range(epoch):
    sess.run(training_step,feed_dict={X:xtrain,Y:ytrain})
    cost_history = np.append(cost_history,sess.run(cost,feed_dict={X: xtrain,Y: ytrain}))

plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,epoch,0,np.max(cost_history)])
plt.show()


pred_y = sess.run(yhat, feed_dict={X: xtest})
#mse = tf.reduce_mean(tf.square(pred_y - ytest))
#print("MSE: %.4f" % sess.run(mse)) 

#fig, ax = plt.subplots()
#ax.scatter(test_y, pred_y)
#ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')
#plt.show()


sess.close()





















