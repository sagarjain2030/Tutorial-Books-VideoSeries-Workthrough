# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 21:35:35 2018

@author: Sagar
"""

import numpy as np
from Model_Representation import IODifferentiation

def numericalModel(X,Y):
    term1 = np.matmul(X.T,X)
    term1_inv = np.linalg.inv(term1)
    
    term2 = np.matmul(term1_inv,X.T)
    theta = np.matmul(term2,Y)
    return theta

def main():
    data = [[2104,5,1,45,460],
            [1416,3,2,40,232],
            [1534,3,2,30,315],
            [852,2,1,36,178]]

    X,Y = IODifferentiation(data)

    numericalModel(X,Y)

if __name__ == "__main__":
    main()

#print(X)
#print(Y)

