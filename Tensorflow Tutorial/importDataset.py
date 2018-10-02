# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:50:17 2018

@author: Sagar
"""


from tensorflow.examples.tutorials.mnist import input_data

def getData():
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    
    # Load data
    X_train = mnist.train.images
    Y_train = mnist.train.labels
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    
    return X_train,X_test,Y_train,Y_test

getData()

