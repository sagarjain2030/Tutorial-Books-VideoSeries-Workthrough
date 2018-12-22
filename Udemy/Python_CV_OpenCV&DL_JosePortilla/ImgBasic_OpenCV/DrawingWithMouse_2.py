# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 12:14:33 2018

@author: Sagar
"""

import cv2
import numpy as np

img = np.zeros((512,512,3))

#True while mouse button down. False while mouse button up
drawing = False
ix,iy = -1,-1

def draw_rectangle(event,x,y,flags,params):
    global ix,iy,drawing
    if(event == cv2.EVENT_LBUTTONDOWN):
        drawing = True
        ix,iy = x,y
    elif(event == cv2.EVENT_MOUSEMOVE):
        if drawing:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
    elif(event == cv2.EVENT_LBUTTONUP):
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)

cv2.namedWindow(winname = 'My_Drawing')
cv2.setMouseCallback('My_Drawing',draw_rectangle)

while True:
    cv2.imshow('My_Drawing',img)
    if(cv2.waitKey(20) & 0xFF == 27):
        break

cv2.destroyAllWindows()