# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 19:50:41 2019

@author: Sagar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_pic(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)

img = cv2.imread('../Data/sammy_face.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
show_pic(img)

edges = cv2.Canny(img,threshold1=127,threshold2=127)
show_pic(edges)

edges = cv2.Canny(img,threshold1=0,threshold2=255)
show_pic(edges)

edges = cv2.Canny(img,threshold1=127,threshold2=255)
show_pic(edges)

#### Deciding Threshold
#### 1.Calculate Median
med_val = np.median(img)
print(med_val)

#### 2.Choosing Lower Threshold -> threshold1
lower = int(max(0,0.7*med_val))

#### 3.Choosing Upper Threshold -> threshold2
upper = int(min(255,1.3*med_val))

edges = cv2.Canny(img,threshold1=lower,threshold2=upper)
show_pic(edges)

blurred = cv2.blur(img,ksize=(5,5))
edges = cv2.Canny(blurred,threshold1=lower,threshold2=upper)
show_pic(edges)

edges = cv2.Canny(blurred,threshold1=lower,threshold2=upper+50)
show_pic(edges)