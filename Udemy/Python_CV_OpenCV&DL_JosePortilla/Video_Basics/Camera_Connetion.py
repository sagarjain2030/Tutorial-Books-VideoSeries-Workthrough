# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:14:51 2019

@author: Sagar
"""

import cv2
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(width)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(height)
print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

#### Passing file video format i.e argument for cv2.VideoWriter_fourcc
### for Windows : DIVX
### for Linux/MacOS : XVID
writer = cv2.VideoWriter('MyVideo.mp4',cv2.VideoWriter_fourcc(*'DIVX'),20,(width,height))
while(True):
    ret, frame = cap.read()
    writer.write(frame)
    cv2.imshow('frame',frame)
    
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',gray)
    
    #BGR2RGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #cv2.imshow('frame',BGR2RGB)
    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
cap.release()
writer.release()
cv2.destroyAllWindows()
    