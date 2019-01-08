# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:52:08 2019

@author: Sagar
"""

import cv2
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)

### Top Left Corner
x = width // 2
y = height // 2

### Width and height of Rectangle
w = width // 4
h = height // 4

### Bottom Right Corner x + w, y + h
 
while(True):
    
    ret,frame = cap.read()
    
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
    cv2.imshow('frame',frame)
    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()