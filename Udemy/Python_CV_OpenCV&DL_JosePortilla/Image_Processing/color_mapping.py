# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 21:25:46 2018

@author: Sagar
"""

import cv2
import matplotlib.pyplot as plt

#print("Showing Original Image")
img = cv2.imread('../Data/00-puppy.jpg')
#plt.imshow(img)

#print("Showing RGB Image")
img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.imshow(img_RGB)

#print("Showing HSV Image")
img_HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#plt.imshow(img_HSV)

print("Showing HLS Image")
img_HLS = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
plt.imshow(img_HLS)

