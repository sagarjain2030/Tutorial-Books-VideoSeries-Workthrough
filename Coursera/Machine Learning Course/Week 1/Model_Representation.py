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

def devCostFunction(i,X,Y_pred,Y):
    m = X.shape[0]
    if(i == 0):
        return ((1/m) * np.sum(np.subtract(Y_pred,Y)))
    else:
        return ((1/m) * np.sum(np.multiply(np.subtract(Y_pred,Y),X))) 

def gradientDescentStep(X,Y,Y_pred,Weight,learning_Rate = 0.01):
    for i in range(len(Weight)):
        Weight[i] = Weight[i] - (learning_Rate * devCostFunction(i,X,Y_pred,Y))
    
    return Weight
    
data = [[2104,460],[1416,232],[1534,315],[852,178]]

#Creating New valid Dataset
# taken W0 = 2 ,W1 = 2
dataset = [[1,4],[3,8],[2,6],[5,12],[8,18]]
def LinearRegression(dataset):
    X,Y = IODifferentiation(dataset)
    Weight = LinearEquation(X)
    print(Weight)
    cost = 10000000
    m = X.shape[0]
    iteration = 0
    #Running Gradien Descent Step
    while(iteration < 1000):
        iteration = iteration + 1
        print("iteration Number: " + str(iteration))
        Y_pred = hypothesisValue(X, Weight)
        cost = calculateCost(Y_pred,Y,m)
        print(cost)
        Weight = gradientDescentStep(X,Y,Y_pred,Weight,learning_Rate = 0.01)
        print(Weight)
    print("Final Weight values are")
    print(Weight)
    print("Final Hypothesis values")
    print(hypothesisValue(X,Weight))

LinearRegression(dataset)
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
#Weight = gradientDescentStep(Weight,learning_Rate=0.1)
print(Weight)