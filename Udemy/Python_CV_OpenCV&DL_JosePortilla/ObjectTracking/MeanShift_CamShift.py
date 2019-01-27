# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:03:24 2019

@author: Sagar
"""

import cv2 
import numpy as np

cap = cv2.VideoCapture(0)
ret,frame = cap.read()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

face_cascade = cv2.CascadeClassifier('../Data/haarcascades/haarcascade_frontalface_default.xml')
face_rect = face_cascade.detectMultiScale(frame)

while(len(face_rect) < 1):
    ret,frame = cap.read()
    face_rect = face_cascade.detectMultiScale(frame)
    print('Searching for Face')
   
(face_x,face_y,w,h) = tuple(face_rect[0])
track_window = (face_x,face_y,w,h)

roi = frame[face_y:face_y+h,face_x:face_x+w]

hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
writer = cv2.VideoWriter('Mean_Track.mp4',cv2.VideoWriter_fourcc(*'DIVX'),20,(width,height))

while(True):
    ret , frame = cap.read()
    if(ret == True):
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        '''
        ###  Mean Shift Tracking
        ret,track_window = cv2.meanShift(dst,track_window,term_crit)
        
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
        ############################################################
        '''
        ### CamShift Tracking
        ret,track_window = cv2.CamShift(dst,track_window,term_crit)
        
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        
        img2 = cv2.polylines(frame,[pts],True,(0,0,255),5)
        ############################################################
        
        writer.write(frame)
        cv2.imshow('face_Reco',frame)
        
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    else:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()