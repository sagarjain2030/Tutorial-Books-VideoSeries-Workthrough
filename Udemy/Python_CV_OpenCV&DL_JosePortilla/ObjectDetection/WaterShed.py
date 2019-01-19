# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 21:28:09 2019

@author: Sagar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

def show_pic(img,cmap='gray'):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)
    
sep_coins = cv2.imread('../Data/pennies.jpg')
show_pic(sep_coins)

### Steps are:
### 1.Median Blurr
sep_blur = cv2.medianBlur(sep_coins,25)
### 2.Grayscale
gray_sep_coins = cv2.cvtColor(sep_blur,cv2.COLOR_BGR2GRAY)
### 3.Binary_Threshold
ret,sep_coins_thre = cv2.threshold(gray_sep_coins,160,255,cv2.THRESH_BINARY_INV)
### 4.Find Contours
sep_coins_thre_copy = copy.deepcopy(sep_coins_thre)
image,contour,hiera = cv2.findContours(sep_coins_thre_copy,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contour)):
    if(hiera[0][i][3] == -1):
        cv2.drawContours(sep_coins,contour,i,(255,0,0),10)
    
show_pic(sep_coins)


#### Watershed Algorithm
img = cv2.imread('../Data/pennies.jpg')
### Blurr Image
img = cv2.medianBlur(img,35)
### Convert to Gray Scale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
### Threshold using OTSU Algorithm
ret,img_thre = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
### Noise Removal
kernel = np.ones((3,3),np.uint8)
img_noise_removal = cv2.morphologyEx(img_thre,cv2.MORPH_OPEN,kernel,iterations=2)
### backgroung Image
sure_bg = cv2.dilate(img_noise_removal,kernel,iterations=3)
### Distance Transform
dist_tran = cv2.distanceTransform(img_noise_removal,cv2.DIST_L2,5)
ret,sure_fg = cv2.threshold(dist_tran,0.7*dist_tran.max(),255,0)
show_pic(sure_fg)
### Finding Unknown Region
sure_fg = np.uint8(sure_fg)
unknow_region = cv2.subtract(sure_bg,sure_fg)
### Marker for Unknown Region
ret,marker = cv2.connectedComponents(sure_fg)
marker = marker + 1
marker[unknow_region==255]=0
### Applying Watershed Algorithm to markers:
marker = cv2.watershed(img,marker)
show_pic(marker)
### Find Contour
image,contour,hiera = cv2.findContours(marker,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contour)):
    if(hiera[0][i][3] == -1):
        cv2.drawContours(sep_coins,contour,i,(255,0,0),10)
    
show_pic(sep_coins)

