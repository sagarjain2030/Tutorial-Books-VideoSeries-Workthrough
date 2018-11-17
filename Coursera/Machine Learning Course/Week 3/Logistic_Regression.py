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
    return 1/(1 + np.exp(-x))

def LogisticRegression(data):
    X,Y = IODifferentiation(data)
    weights = paramintialization(X)
    exponentValue = (X*weights.T)
    print(exponentValue)
    print(calculateSigmoid(exponentValue))
    
def main():
    data = [[2104,5,1,45,1],
            [1416,3,2,40,0],
            [1534,3,2,30,1],
            [852,2,1,36,0]]

    LogisticRegression(data)

    
if __name__ == "__main__":
    main()