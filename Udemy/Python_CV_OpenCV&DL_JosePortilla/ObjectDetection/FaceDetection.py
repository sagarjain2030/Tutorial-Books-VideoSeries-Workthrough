# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:44:43 2019

@author: Sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_pic(img,cmap='gray'):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)

nadia = cv2.imread('../Data/Nadia_Murad.jpg',0)
Dennis = cv2.imread('../Data/Denis_Mukwege.jpg',0)
solvay = cv2.imread('../Data/solvay_conference.jpg',0)

show_pic(nadia)
show_pic(Dennis)
show_pic(solvay)

face_cascade = cv2.CascadeClassifier('../Data/haarcascades/haarcascade_frontalface_default.xml')

def face_Detect(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img)
    
    for (x,y,w,h) in face_rect:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    
    return face_img

result = face_Detect(Dennis)
show_pic(result)

result = face_Detect(nadia)
show_pic(result)

result = face_Detect(solvay)
show_pic(result)
    
def adjusted_face_Detect(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img,1.2,3)
    
    for (x,y,w,h) in face_rect:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    
    return face_img

result = adjusted_face_Detect(solvay)
show_pic(result)

eye_cascade = cv2.CascadeClassifier('../Data/haarcascades/haarcascade_eye.xml')

def eye_detect(img):
    eye_img = img.copy()
    eye_rect = eye_cascade.detectMultiScale(eye_img)
    
    for (x,y,w,h) in eye_rect:
        cv2.rectangle(eye_img,(x,y),(x+w,y+h),(255,255,255),10)
    
    return eye_img

result = eye_detect(nadia)
show_pic(result)

result = eye_detect(Dennis)
show_pic(result)

vid = cv2.VideoCapture(0)

while True:
    ret,frame = vid.read(0)
    
    frame = face_Detect(frame)
    cv2.imshow('Video Face Detect',frame)
    
    k = cv2.waitKey(1)
    if k == 27:
        break

vid.release()
cv2.destroyAllWindows()