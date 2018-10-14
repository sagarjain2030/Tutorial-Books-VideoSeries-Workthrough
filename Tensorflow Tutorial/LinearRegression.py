# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:55:39 2018

@author: Sagar
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


W_in = 3.14
b_in = 5.6

XX = []
YY = []

for i in range(100):
    c = np.random.rand()
    d = W_in * c + b_in
    XX.append(c)
    YY.append(d)

train_X = np.asarray(XX)
train_Y = np.asarray(YY)

XXX = []
YYY = []
for i in range(128,158):
    c = np.random.rand()
    d = W_in * c + b_in
    XXX.append(c)
    YYY.append(d)

test_X = np.asarray(XXX)
test_Y = np.asarray(YYY)

n_sample = train_X.shape[0]
print(n_sample) 

epochNumbers = 2000
learningRate = 0.001

X = tf.placeholder(tf.float32,name="X")
Y = tf.placeholder(tf.float32,name="Y")

W = tf.Variable(np.random.rand(), name="weight")
b = tf.Variable(np.random.rand(), name="bias")

pred = tf.add(tf.multiply(W,X),b)
cost = tf.reduce_sum(tf.pow((pred - Y),2))/(n_sample)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learningRate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("Randomly initialized ")
    print(sess.run(W))
    print(sess.run(b))
    for epoch in range(epochNumbers):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        c = sess.run(cost,feed_dict={X: train_X, Y:train_Y})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),"W=", sess.run(W), "b=", sess.run(b))    
 
    print("optimizer finished")
    
    training_cost = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
    print("Training cost is " + str(training_cost) + "  with W = " + str(sess.run(W)) + "and b = " + str(sess.run(b)))
    
    #testCostFunc = tf.reduce_sum(tf.pow(pred-Y,2))/test_X.shape[0]
    testing_cost = sess.run(cost, feed_dict={X:test_X,Y:test_Y})
    print("Testing cost is " + str(testing_cost))
    
    plt.plot(train_X, train_Y, 'go', label='training data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b) , 'r^' , label='trainFitted Data')
    plt.legend()
    plt.show()

    plt.plot(test_X, test_Y, 'go', label='testing data')
    plt.plot(test_X, sess.run(W) * test_X + sess.run(b) , 'r^' , label='testFitted Data')
    plt.legend()
    plt.show()    
