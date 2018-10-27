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
    W = np.random.rand(X.shape[1]+1)
    return W

def hypothesisValue(X,weights):
    Y_pred = []
    W = np.array(weights)[1:]
    b = np.array(weights)[:1]
    Y_pred.append(np.multiply(W.T,X) + b)
    return Y_pred


def calculateCost(Y_pred,Y,m):
    cost = (1/ (2*m)) * np.sum(np.power(np.subtract(Y_pred,Y),2))
    return cost

def devCostFunction(i):
    #For the time being since haven't calculated derivatives yet
    return 1

def gradientDescentStep(Weight,learning_Rate = 0.01):
    for i in range(len(Weight)):
        Weight[i] = Weight[i] - (learning_Rate * devCostFunction(i))
    
    return Weight
    
data = [[2104,460],[1416,232],[1534,315],[852,178]]

X,Y = IODifferentiation(data)
Weight = LinearEquation(X)
m = X.shape[0]
print(X)
print(Y)
print(X.shape)
print(Y.shape)

print(Weight)


print()
print('************************************************************************')
print("Y     = W0           +    X   *  W1")
for i in range(X.shape[0]):
    print(str(Y[i]) + " = " + str(Weight[0]) +  " + " + str(X[i]) +  " * "  + str(Weight[1]))
print('************************************************************************')

Y_pred = hypothesisValue(X,Weight)
print(Y_pred)
cost = calculateCost(Y_pred,Y,m)
print(cost)
Weight = gradientDescentStep(Weight,learning_Rate=0.1)
print(Weight)