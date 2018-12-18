# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:44:30 2018

@author: Sagar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#print("Blank Image")
blank_img = np.zeros(shape=(512,512,3),dtype=np.int16)
#plt.imshow(blank_img)


## Drawing Rectangle on image
#print("Drawing Rectangle 1")
cv2.rectangle(blank_img,pt1=(384,10),pt2=(500,150),color=(0,255,0),thickness=10)
#plt.imshow(blank_img)


## Drawing Rectangle at center
#print("Drawing Rectangle 2")
cv2.rectangle(blank_img,pt1=(200,200),pt2=(300,300),color=(0,0,255),thickness=10)
#plt.imshow(blank_img)

## Drawing Circle
#print("Drawing Circle")
cv2.circle(img=blank_img,center=(100,100),radius=50,color=(255,0,0),thickness=10)
#plt.imshow(blank_img)

## Drawing filled Circle
#print("Drawing Circle")
cv2.circle(img=blank_img,center=(400,400),radius=50,color=(255,0,0),thickness=-1)
#plt.imshow(blank_img)

## Drawing Line
#print("Drawing line")
cv2.line(img=blank_img,pt1=(0,0),pt2=(512,512),color=(102,255,255),thickness = 5)
#plt.imshow(blank_img)

#print("Showing Text")
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_img,"Hello",org=(10,500),fontFace=font,fontScale=4,color=(255,255,255),thickness=3,lineType=cv2.LINE_AA)
#plt.imshow(blank_img)

## Drawing Polygon
## Create a list of vertices
## reshape the vertices with (-1,1,2)
## pass points as list in polylines

print("Drawing polygon")
blank_img = np.zeros(shape=(512,512,3),dtype=np.int32)
vertices = np.array([[100,300], [200,200], [400,300], [200,400] ],dtype=np.int32)
pts = vertices.reshape(-1,1,2)
cv2.polylines(blank_img,[pts],isClosed = True,color=(255,0,0),thickness=5)
plt.imshow(blank_img)