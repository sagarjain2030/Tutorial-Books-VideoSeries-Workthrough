# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:18:37 2019

@author: Sagar
"""

import cv2
import time 

pt1 = (0,0)
pt2 = (0,0)
topleft_clicked = False
botRight_clicked = False

### CALLBACK FUNCTION RECTANGLE
def draw_rectangle(event,x,y,flags,params):
    global pt1,pt2,topleft_clicked,botRight_clicked
    
    if(event == cv2.EVENT_LBUTTONDOWN):
        
        ### RESET LOGIC
        if topleft_clicked ==True & botRight_clicked == True:
            pt1 = (0,0)
            pt2 = (0,0)
            topleft_clicked = False
            botRight_clicked = False
            
        if topleft_clicked == False:
            pt1 = (x,y)
            topleft_clicked = True
        
        elif botRight_clicked == False:
            pt2 = (x,y)
            botRight_clicked = True

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)

cv2.namedWindow('MyCallback')
cv2.setMouseCallback('MyCallback',draw_rectangle)

while(True):
    
    ret,frame = cap.read()
    
    if topleft_clicked:
        cv2.circle(frame,center=pt1,radius=5,color=(0,255,0),thickness=-1)
    if topleft_clicked & botRight_clicked:
        cv2.rectangle(frame,pt1,pt2,(0,255,0),3)
        
    
    cv2.imshow('MyCallback',frame)    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()