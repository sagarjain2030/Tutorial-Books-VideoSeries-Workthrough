# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:54:39 2018

@author: Sagar
"""

import numpy as np

def IODifferentiation(data):
    for i in range(len(data)):
        data[i].insert(0,1)
    data = np.asmatrix(data)    
    X = data[:,:-1]
    Y = data[:,-1:]
    return X,Y

def LinearEquationoefficient(X):
    W = np.random.rand(X.shape[1])
    W = np.matrix(W)
    return W

def calculateError(X,Weight,Y):
    error = (X*Weight.T) - Y
    return error

def calculateCost(Y_pred,Y,m):
    cost = (1/ (2*m)) * np.sum(np.power(np.subtract(Y_pred,Y),2))
    return cost


def ComputeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def GradientDescent(X,Y,Weight,iterations,learning_Rate = 0.01):
    newWeight = np.matrix(np.zeros(Weight.shape))
    cost = np.zeros(iterations)
    for i in range(iterations):
        error = calculateError(X,Weight,Y)
        
        for j in range(X.shape[1]):          
            multiplyingFactor = np.multiply(error,X[:,j])

            newWeight[0,j] = Weight[0,j] -  ((learning_Rate/len(X)) * np.sum(multiplyingFactor)) 
        Weight = newWeight
        cost[i] = ComputeCost(X,Y,Weight)
        print("Cost for iteration " + str(i))
        print(Weight)
        print(cost[i])
        
    return Weight,cost

#Creating New valid Dataset
# taken W0 = 2 ,W1 = 2,W2 = 2
data = [[1,1,6],[3,4,16],[2,5,16],[5,7,26],[8,3,24]]

X,Y = IODifferentiation(data)
Weight = LinearEquationoefficient(X)
Weight,cost = GradientDescent(X,Y,np.matrix(Weight),iterations = 1000,learning_Rate = 0.01)
print(Weight,cost)