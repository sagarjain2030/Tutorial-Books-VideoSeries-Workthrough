# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:13:09 2019

@author: Sagar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_pic(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')

flat_chess = cv2.imread('../Data/flat_chessboard.png')
RGB_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)
show_pic(RGB_flat_chess)

flat_chess = cv2.imread('../Data/flat_chessboard.png')
gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2GRAY)
show_pic(flat_chess)

real_chess = cv2.imread('../Data/real_chessboard.jpg')
RGB_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)
show_pic(real_chess)

real_chess = cv2.imread('../Data/real_chessboard.jpg')
gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)
show_pic(real_chess)

gray  = np.float32(gray_flat_chess)
dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)

dst = cv2.dilate(dst,None)
RGB_flat_chess[dst > 0.01*dst.max()] = [255,0,0]
show_pic(RGB_flat_chess)

gray  = np.float32(gray_real_chess)
dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)
dst = cv2.dilate(dst,None)
RGB_real_chess[dst > 0.01*dst.max()] = [255,0,255]
show_pic(RGB_real_chess)

flat_chess = cv2.imread('../Data/flat_chessboard.png')
RGB_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)
gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2GRAY)

real_chess = cv2.imread('../Data/real_chessboard.jpg')
RGB_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)
gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray_flat_chess,0,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y=i.ravel()
    cv2.circle(RGB_flat_chess,(x,y),3,(255,0,0),-1)
    
show_pic(RGB_flat_chess)


corners = cv2.goodFeaturesToTrack(gray_real_chess,100,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y=i.ravel()
    cv2.circle(RGB_real_chess,(x,y),3,(255,0,0),-1)
    
show_pic(RGB_real_chess)