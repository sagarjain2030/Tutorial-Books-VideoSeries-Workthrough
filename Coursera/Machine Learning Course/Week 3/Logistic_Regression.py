# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:00:13 2018

@author: Sagar
"""

import numpy as np
import math

def IODifferentiation(data):
    for i in range(len(data)):
        data[i].insert(0,1)
    data = np.asmatrix(data)    
    X = data[:,:-1]
    Y = data[:,-1:]
    return X,Y

def paramintialization(X):
    W = np.random.rand(X.shape[1])
    W = np.matrix(W)
    return W

def calculateSigmoid(x):
    return 1.0/(1 + np.exp(-x))

def loss(h, y):
    return (-y.T * np.log(h) - (1 - y).T * np.log(1 - h)).mean()

def calculateCost(x,Y):
    x = x + np.exp(-10)
    print(np.log(x))
    print(np.log(1 - x))
    #cost = (-1) * np.multiply(Y,np.log(x)) + (-1) * np.multiply((1-Y),np.log(1-))


def LogisticRegression(data):
    X,Y = IODifferentiation(data)
    weights = paramintialization(X)
    exponentValue = np.dot(X,weights.T)
    h = calculateSigmoid(exponentValue)
    print(loss(h,Y))
    
def main():
    data = [[21,5,1,45,1],
            [14,3,2,40,0],
            [15,3,2,30,1],
            [8,2,1,36,0]] 
    
    LogisticRegression(data)

    
if __name__ == "__main__":
    main()