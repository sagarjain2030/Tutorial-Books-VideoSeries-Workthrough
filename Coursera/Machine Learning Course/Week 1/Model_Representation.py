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

def LinearEquation(X):
    Weight = []
    for i in range(X.shape[0]):
        W = np.random.rand(X.shape[1]+1)
        Weight.append(list(W))
    return Weight

def hypothesisValue(X,weights):
    Y_pred = []
    W = np.array(weights)[:,1:]
    b = np.array(weights)[:,:1]
    Y_pred.append(np.matmul(W.T,X) + b)
    return Y_pred

data = [[2104,460],[1416,232],[1534,315],[852,178]]

X,Y = IODifferentiation(data)
Weight = LinearEquation(X)

print(X)
print(Y)
print(X.shape)
print(Y.shape)

print(Weight)


print()
print('************************************************************************')
print("Y     = W0           +    X   *  W1")
for i in range(X.shape[0]):
    print(str(Y[i]) + " = " + str(Weight[i][0]) +  " + " + str(X[i]) +  " * "  + str(Weight[i][1]))
print('************************************************************************')

Y_pred = hypothesisValue(X,Weight)
