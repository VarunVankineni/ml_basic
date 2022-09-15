# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:44:23 2018

@author: Varun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
"""
Setting working device to CPU as my GPU got fried 
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

"""
Importing Data with flattened images
"""
data = input_data.read_data_sets("MNIST_data/")
X_train, y_train           = data.train.images, data.train.labels
X_test, y_test             = data.test.images, data.test.labels
X_valid, y_valid             = data.validation.images, data.validation.labels

"""
Normalized scaling to (0,1)
"""
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

"""
Reshaping to 28x28 images
"""
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
X_valid = X_valid.reshape(X_valid.shape[0],28,28,1)

"""
Padding with zeros, images are now 32x32
"""
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_valid = np.pad(X_valid, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

"""
placeholders and one hot encoding for cross entropy calculation
"""
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
onehoty = tf.one_hot(y, 10)

"""
%%PARAMETERS
"""
filters1 = 6           #number of filters in 1st convolution
filterframe = [5,5]    #frame of the convolution filters
filters2 = 16          #number of filters in 2nd convolution
maxpoolframe = [2,2]   #maxpooling frame size
maxpoolstrides = [2,2] #stride of maxpooling frame
fconshape1 = 120       #neurons in 1st fully connected layer
fconshape2 = 84        #neurons in 2nd fully connected layer 
classes = 10           #final neurons for logits  
a = 0.1                #learning rate 
maxitr = 3             #total number of iterations over the data set 
batchsize = 128        #batchsize for each gradient descent run


"""
Network Architecture 
"""
conv1 = tf.layers.conv2d(x, filters1, filterframe, activation = tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1,maxpoolframe,maxpoolstrides)
conv2 = tf.layers.conv2d(pool1, filters2, filterframe, activation = tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2,maxpoolframe,maxpoolstrides)
fcon0 = flatten(pool2)
fcon1 = tf.layers.dense(fcon0,fconshape1,tf.nn.relu)
fcon2 = tf.layers.dense(fcon1,fconshape2,tf.nn.relu)
logits = tf.layers.dense(fcon2, classes)

"""
Loss function and optimizer algorithm definition
"""
entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = onehoty)
loss = tf.reduce_mean(entropy)
algo = tf.train.GradientDescentOptimizer(a).minimize(loss)

"""
Predictions and accuracy helper function
"""

checkprediction = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(onehoty, 1))
def calc_accuracy(X,Y,batchsize):
    total_accurate = 0
    for start in range(0,X.shape[0],batchsize):
        end = start+batchsize
        accurate = np.sum(sess.run(checkprediction, feed_dict = {x:X[start:end], y:Y[start:end]}))
        total_accurate += accurate
    return  100*total_accurate/X.shape[0]

"""
Intitiate session
"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())

"""
Batch gradient descent
"""
for i in range(maxitr):
    for start in range(0,X_train.shape[0],batchsize):
        end = start+batchsize
        sess.run(algo,feed_dict = {x:X_train[start:end],y:y_train[start:end]})

    print(i,calc_accuracy(X_train,y_train,batchsize))
    
"""
Calculating final prediction accuracies and printing the same
"""
accuracy = calc_accuracy(X_train,y_train,batchsize)
accuracy_v =calc_accuracy(X_valid,y_valid,batchsize) 
accuracy_t =calc_accuracy(X_test,y_test,batchsize) 
print(accuracy,accuracy_v,accuracy_t)