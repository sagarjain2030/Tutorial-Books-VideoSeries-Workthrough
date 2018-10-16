# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 19:03:37 2018

@author: Sagar
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/",one_hot=True)

X = tf.placeholder(tf.float32,[None,28*28])
Y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.random_normal([10]))

learnRate = 0.01
epochNumbers = 50
batchSize = 128

pred = tf.matmul(X,W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learnRate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochNumbers):
        avgCost = 0
        totalbatch = int(mnist.train.num_examples/batchSize)
        
        for i in range(totalbatch):
            batchX, batchY = mnist.train.next_batch(batchSize)
            opti = sess.run(optimizer,feed_dict={X:batchX,Y:batchY})
            c = sess.run(cost,feed_dict={X:batchX,Y:batchY})
            avgCost += c/totalbatch
        
        print("Epoch:"+ str(epoch) + " cost=", str(avgCost))
    
    print("Optimiztion Finished")
    
    correctPrediction = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction,tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))