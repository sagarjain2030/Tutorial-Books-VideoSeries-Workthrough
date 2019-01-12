# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 19:37:13 2019

@author: Sagar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_pic(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)

img = cv2.imread('../Data/internal_external.png',0)
show_pic(img,'gray')

img_contour,contour,hiera = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
external_Contour = np.zeros(img.shape)

for i in range(len(contour)):
    ### External Contour = -1
    if hiera[0][i][3] == -1:
        cv2.drawContours(external_Contour,contour,i,255,-1)

show_pic(external_Contour,'gray')

internal_Contour = np.zeros(img.shape)
for i in range(len(contour)):
    ### Internal Contour != -1
    if hiera[0][i][3] != -1:
        cv2.drawContours(internal_Contour,contour,i,255,-1)

show_pic(internal_Contour,'gray')

all_contour = np.zeros(img.shape)
for i in range(len(contour)):
    ### Internal Contour != -1
    cv2.drawContours(all_contour,contour,i,255,1)

show_pic(all_contour,'gray')
