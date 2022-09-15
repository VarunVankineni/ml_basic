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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import style
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


data = input_data.read_data_sets("MNIST_data/")
X_train, y_train           = data.train.images, data.train.labels
X_test, y_test             = data.test.images, data.test.labels
X_valid, y_valid             = data.test.images, data.test.labels

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
X_valid = X_valid.reshape(X_valid.shape[0],28,28,1)


X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_valid = np.pad(X_valid, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
onehoty = tf.one_hot(y, 10)

batchsize = 1



mu = 0
sigma = 0.01
filshape1 = [5,5,1,6]
filshape2 = [5,5,6,16]
maxpoolframe = [1,2,2,1]
maxpoolstrides = [1,2,2,1]

fconshape1 = 120
fconshape2 = 84


a = 0.4
maxitr = 1
batchsize = 50


def weights(shape):
    return tf.Variable(tf.random_normal(shape = shape,mean = mu,stddev = sigma))
def bias(shape):
    return tf.Variable(tf.zeros(shape[-1]))

def convlayer(data,w,b,padding='VALID',relu = 1):
    conv = tf.nn.conv2d(data,w,strides = [1,1,1,1],padding = padding)+b
    if(relu==1):
        conv = tf.nn.relu(conv)
    
    return conv

def maxpool(conv,frame=maxpoolframe,strides=maxpoolstrides,padding='VALID'):
    pool = tf.nn.max_pool(conv,ksize=frame,strides = strides,padding = padding)
    return pool

def fullcon(fcon,w,b,relu = 1):
    fconl = tf.matmul(fcon,w)+b
    if(relu==1):
        fconl = tf.nn.relu(fconl)
    return fconl



w1 = weights(filshape1)
b1 = bias(filshape1)
conv1 = convlayer(x,w1,b1) 
pool1 = maxpool(conv1)


w2 = weights(filshape2)
b2 = bias(filshape2)
conv2 = convlayer(pool1,w2,b2)
pool2 = maxpool(conv2)

fcon0 = flatten(pool2)

w3 = weights([fcon0.shape.as_list()[1],fconshape1])
b3 = bias([fcon0.shape.as_list()[1],fconshape1])
fcon1 = fullcon(fcon0,w3,b3) 

w4 = weights([fcon1.shape.as_list()[1],fconshape2])
b4 = bias([fcon1.shape.as_list()[1],fconshape2])
fcon2 = fullcon(fcon1,w4,b4) 

w5 = weights([fcon2.shape.as_list()[1],10])
b5 = bias([fcon2.shape.as_list()[1],10])
logits = fullcon(fcon2,w5,b5,relu=0)

entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = onehoty)
loss = tf.reduce_mean(entropy)
algo = tf.train.GradientDescentOptimizer(learning_rate = a).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(maxitr):
    for start in range(0,X_train.shape[0],batchsize):
        end = start+batchsize
        sess.run(algo,feed_dict = {x:X_train[start:end],y:y_train[start:end]})
"""
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(onehoty, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for start in range(0,X_valid.shape[0],batchsize):
    end = start+batchsize
    accuracy = sess.run(accuracy_operation,feed_dict = {x:X_valid[start:end],y:y_valid[start:end]})
    accuracy += (accuracy*X_valid[start:end].shape[0])

accuracy = accuracy/X_valid.shape[0]    
"""
afterimage = sess.run(logits, feed_dict = {x : X_train[506:547]})






























