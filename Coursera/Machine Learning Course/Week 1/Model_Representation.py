# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:54:39 2018

@author: Sagar
"""

import numpy as np

def IODifferentiation(data):
    data = np.asarray(data)
    X = data[:,:-1]
    Y = data[:,-1:]
    
    return X,Y

def LinearEquation(X,Y):
    W0 = np.random.rand(X.shape[0],X.shape[1])
    W1 = np.random.rand(X.shape[0],X.shape[1])
    return (W0,W1)

data = [[2104,460],[1416,232],[1534,315],[852,178]]

X,Y = IODifferentiation(data)
tu = LinearEquation(X,Y)
print(X)
print(Y)
print(X.shape)
print(Y.shape)

print()
print('************************************************************************')
print("Y     = W0           +    X   *  W1")
for i in range(X.shape[0]):
    print(str(Y[i]) + " = " + str(tu[0][i]) +  " + " + str(X[i]) +  " * "  + str(tu[1][i]))
print('************************************************************************')
