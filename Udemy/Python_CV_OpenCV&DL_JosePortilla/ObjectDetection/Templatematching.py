# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:44:07 2019

@author: Sagar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy


def show_pic(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)

full = cv2.imread('../Data/sammy.jpg')
full = cv2.cvtColor(full,cv2.COLOR_BGR2RGB)
show_pic(full)

face = cv2.imread('../Data/sammy_face.jpg')
face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
show_pic(face)

#### You must have exactly same image which you are searching into larger image
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for m in methods:
    full_copy = copy.deepcopy(full)
    method = eval(m)
    
    res = cv2.matchTemplate(full_copy,face,method)
    
    #### Get the location from Heat Map generated
    min_val,max_val,min_val_loc,max_val_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_val_loc
    else:
        top_left = max_val_loc
    
    width,height,channels = face.shape
    
    bottom_right = (top_left[0] + width,top_left[1]+height)
    
    cv2.rectangle(full_copy,top_left,bottom_right,color=(0,0,255),thickness=10)
    plt.subplot(121)
    plt.imshow(res)
    plt.title('HeatMap for Template Matching')
    
    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('Detection of Template')
    
    plt.suptitle(m)
    plt.show()
    
    
    print()
    print()
    