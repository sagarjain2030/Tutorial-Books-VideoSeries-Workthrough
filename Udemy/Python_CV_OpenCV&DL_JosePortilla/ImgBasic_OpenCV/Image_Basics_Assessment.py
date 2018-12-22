# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 14:13:18 2018

@author: Sagar
"""

import cv2

img = cv2.imread('..\Data\dog_backpack.jpg')

def draw_circle(event,x,y,flags,params):
    if(event == cv2.EVENT_RBUTTONDOWN):
        cv2.circle(img,(x,y),100,(0,0,255),10)

cv2.namedWindow(winname = 'Image')
cv2.setMouseCallback('Image',draw_circle)
    

while True:
    cv2.imshow('Image',img)
    if(cv2.waitKey(20) & 0xFF == 27):
        break
cv2.destroyAllWindows()
