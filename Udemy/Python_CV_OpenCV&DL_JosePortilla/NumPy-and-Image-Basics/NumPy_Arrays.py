# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:22:21 2018

@author: Sagar
"""

import numpy as np

mylist = [1,2,3]
#print(type(mylist))

myArray = np.array(mylist)
#print(type(myArray))

zeroTo10 = np.arange(0,10)
#print(zeroTo10)

zeroto10by2 = np.arange(0,10,2)
#print(zeroto10by2)

zero5 = np.zeros(shape=(5,5))
#print(zero5)

ones4 = np.ones((2,4))
#print(ones4)

np.random.seed(101)
arr = np.random.randint(0,100,10)
#print(arr)

arr2 = np.random.randint(0,100,10)
#print(arr2)

#### Printing Max, Min and Average with index value
#print(arr.max())
#print(arr.argmax())

#print(arr.min())
#print(arr.argmin())

#print(arr.mean())


#print(arr.shape)

arr_1 = arr.reshape((2,5))
#print(arr_1)

arr_2 = arr.reshape((5,2))
#print(arr_2)

mat = np.arange(0,100).reshape((10,10))
print(mat)

row = 0
col = 1

print(mat[row,col])
print(mat[4,6])
print(mat[:,1])

temp = mat[2,:]
temp = temp.reshape((10,1))
print(temp)

print(mat[0:3,0:3])

myNewMat = mat.copy()
myNewMat[0:3,0:4] = 0
print(myNewMat)

myNewMat[0:6,:] = 999
print(myNewMat)