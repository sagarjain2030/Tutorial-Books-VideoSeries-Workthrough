# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:35:13 2019

@author: Sagar
"""

import cv2
import time

cap = cv2.VideoCapture('MyVideo.mp4')
FPS = cap.get(cv2.CAP_PROP_FPS)

if(cap.isOpened == False):
    print('Error. File not found OR Wrong CODEC Used')

while(cap.isOpened):
    ret,frame = cap.read()
    if ret == True:
        cv2.imshow('frame',frame)
        time.sleep(1/FPS) ### To launch video in human readable format
    
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
    