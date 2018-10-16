# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:29:47 2018

@author: Sagar
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/",one_hot=True)

TrainingSize = 5000
TestingSize = 200
accuracy = 0

X_train, Y_train = mnist.train.next_batch(TrainingSize) 
X_test, Y_test = mnist.test.next_batch(TestingSize) #200 for testing

Train_X = tf.placeholder(tf.float32,[None,784])
Test_X = tf.placeholder(tf.float32,[784])

#Using L1 distance method:
distance = tf.reduce_sum(tf.abs(tf.subtract(Train_X,Test_X)),reduction_indices=1)
pred = tf.arg_min(distance,0)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(len(X_test)):
        predIndex = sess.run(pred,feed_dict={Train_X: X_train,Test_X: X_test[i,:]})
        print("Test", i, " Prediction: ", np.argmax(Y_train[predIndex]),"True Class:", np.argmax(Y_test[i]))
        
        if(np.argmax(Y_train[predIndex]) == np.argmax(Y_test)):
            accuracy += 1./len(X_test)
    
    print("Done!")
print("Accuracy:", accuracy*100)