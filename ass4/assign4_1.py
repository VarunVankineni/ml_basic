# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:09:36 2018

@author: Aravinth C K
"""



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# mnist.train.images
# mnist.train.labels

#Computation Graph

num_features = 84
num_labels = 10
learning_rate = 0.2
batch_size = 5500
num_steps = 30


train_x_temp = tf.placeholder(tf.float32)
train_y = tf.placeholder(tf.float32,[None,10])
#train_x_temp2 = tf.reshape(train_x_temp, [1000,28,28])
train_x = tf.reshape(train_x_temp, [-1,28,28,1])
#train_y = tf.reshape(train_y_temp, [-1,1,10])

layer1 = tf.layers.conv2d(inputs = train_x, filters = 6, kernel_size = [5,5], padding = "valid", 
                          activation = tf.nn.relu, use_bias = True, trainable = True)
max_pool1 = tf.layers.max_pooling2d(inputs=layer1, pool_size=[2, 2], strides=2)
layer2 = tf.layers.conv2d(inputs = max_pool1, filters = 16, kernel_size = [5,5], padding = "valid", 
                          activation = tf.nn.relu, use_bias = True, trainable = True)
max_pool2 = tf.layers.max_pooling2d(inputs=layer2, pool_size=[2, 2], strides=2)
flat1 = tf.reshape(max_pool2, [-1, 4 *4 * 16])
fc1 = tf.layers.dense(inputs=flat1, units=120, activation=tf.nn.relu, use_bias = True, trainable = True)
fc2 = tf.layers.dense(inputs=fc1, units=84, activation=tf.nn.relu, use_bias = True, trainable = True)

logits = tf.layers.dense(inputs=fc2, units=10, use_bias = True, trainable = True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_y, logits=logits))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

train_prediction = tf.nn.softmax(logits)
valid_prediction = tf.nn.softmax(logits)

    
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu
     
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    for epoch in range(num_steps):
        
        for i in range(int(mnist.train.num_examples/batch_size)):
            train_batch_x, train_batch_y = mnist.train.next_batch(batch_size)
            fd = {train_x_temp : train_batch_x, train_y : train_batch_y}    
            _, cost, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=fd)
 
        print("Epoch", epoch )
        
        v_pred = valid_prediction.eval(feed_dict = {train_x_temp : mnist.validation.images, train_y : mnist.validation.labels})
        
        print("Validation accuracy:", accuracy(v_pred, mnist.validation.labels)) 
        
    t_pred = valid_prediction.eval(feed_dict = {train_x_temp : mnist.test.images, train_y : mnist.test.labels})    
    print("Test accuracy: ", accuracy(t_pred, mnist.test.labels))

    
    
    
