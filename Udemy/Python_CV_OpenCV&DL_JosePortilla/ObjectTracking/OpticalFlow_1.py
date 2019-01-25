# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 22:23:14 2019

@author: Sagar
"""

import numpy as np
import cv2

corner_track_params = dict(maxCorners = 10, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
lk_params = dict(winSize=(200,200), maxLevel = 2, criteria= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,0.03))

cap = cv2.VideoCapture(0)
_, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    
#ret, prev_frame =  cap.read()
#prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_RGB2GRAY)

prev_Pts  = cv2.goodFeaturesToTrack(prev_gray,mask=None,**corner_track_params)
mask = np.zeros_like(prev_frame)

while(True):
    ret ,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_Pts, None, **lk_params)
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_Pts, None, **lk_params)
    good_new = nextPts[status==1]
    good_prev = prev_Pts[status==1]
    
    for i,(new,prev) in enumerate(zip(good_new,good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        mask = cv2.line(mask,(x_new,y_new),(x_prev,y_prev),(0,255,0),3)
        frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1)
        
    img = cv2.add(frame,mask)
    cv2.imshow('tracking',img)
    
    k = cv2.waitKey(30) & 0xFF
    if k == ord('q'):
        break;
    
    prev_gray = gray_frame.copy()
    prev_Pts = good_new.reshape(-1,1,2)
    
cap.release()
cv2.destroyAllWindows()
    


        

