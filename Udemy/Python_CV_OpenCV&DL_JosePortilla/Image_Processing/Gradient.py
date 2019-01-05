# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 10:44:15 2019

@author: Sagar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
   
print("Image Listing")
print('1 ---->  Original Image\n2 ---->  Sobel Horizontal operator\n\
3 ---->  Sobel Vertical operator\n4 ---->  Sobel Both Axes operator\n\
5 ---->  Sobel laplacian operator\n6 ---->  Blending SobelX and SobelY\n\
7 ---->  Thresholding on Blended Image\n8,9 ---->  Morphological operation on Threshold Image\n\
10 ---->  Adding Morphed Images\n11 ---->  Gradient Morphing on blend Image\n')
img = cv2.imread('../data/sudoku.jpg',0)
display_img(img)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
display_img(sobelx)

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
display_img(sobely)

### dx +dy must be > 0 and dx >=0 and dy >= 0.So don't use 0,0
sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)
display_img(sobelxy)

### Laplacian Gradient
laplacian = cv2.Laplacian(img,cv2.CV_64F)
display_img(laplacian)

blended = cv2.addWeighted(src1=sobelx,alpha=0.5,src2=sobely,beta=0.5,gamma=0)
display_img(blended)

ret,th1 = cv2.threshold(blended,125,255,cv2.THRESH_BINARY)
display_img(th1)

kernel = np.ones((3,3),dtype=np.uint8)
closed = cv2.morphologyEx(th1,cv2.MORPH_CLOSE,kernel)
display_img(closed)

opened = cv2.morphologyEx(th1,cv2.MORPH_OPEN,kernel)
display_img(opened)

result = cv2.addWeighted(src1=opened,alpha=0.5,src2=closed,beta=0.5,gamma=0)
display_img(result)

kernel = np.ones((4,4),dtype=np.uint8)
morphGradient = cv2.morphologyEx(blended,cv2.MORPH_GRADIENT,kernel)
display_img(morphGradient)

                         