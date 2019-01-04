# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 20:06:53 2019

@author: Sagar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_pic(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    
def load_img():
    blank_img = np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, text='ABCDE', org=(50,300), fontFace=font, fontScale=5, color=(255,255,255), thickness=25, lineType=cv2.LINE_AA)
    return blank_img

img = load_img()
show_pic(img)

### Erosion
print('Erosion Kernel Iteration Number = 1')
kernel = np.ones((5,5),dtype=np.uint8)
result = cv2.erode(img,kernel,iterations=1)
show_pic(result)

print('Erosion Kernel Iteration Number = 4')
kernel = np.ones((5,5),dtype=np.uint8)
result = cv2.erode(img,kernel,iterations=4)
show_pic(result)

### Openings: Erosion followed by Dilation, Best for background noise
print("Image of White Noise")
white_noise = np.random.randint(low=0,high=2,size=(600,600)) * 255
show_pic(white_noise)
print("Image with White Noise")
noise_img = white_noise + img
show_pic(noise_img)

print("opening used on white noise image")
opening = cv2.morphologyEx(noise_img,cv2.MORPH_OPEN,kernel)
show_pic(opening)

print('Next 3 images are same as above 3 except kenel is of size 3')
kernel = np.ones((3,3),dtype=np.uint8)
white_noise = np.random.randn(600,600) * 255
show_pic(white_noise)
noise_img = white_noise + img
show_pic(noise_img)

opening = cv2.morphologyEx(noise_img,cv2.MORPH_OPEN,kernel)
show_pic(opening)

#### Closing:Dialtion followed by Erosion, Best for foreground noise
print("Image of black Noise i.e negative noise")
black_noise = np.random.randint(low=0,high=2,size=(600,600)) * -255
show_pic(black_noise)
black_noise_img = black_noise + img
black_noise_img[black_noise_img==-255] = 0

print("Image with black Noise. -255 made to zero")
show_pic(black_noise_img)

print("closing used on black noise image")
kernel = np.ones((5,5),dtype=np.uint8)
closing =  cv2.morphologyEx(black_noise_img,cv2.MORPH_CLOSE,kernel)
show_pic(closing)

### Gradient: Difference taken between dilation and erosion
### One of the method for Edge Detection
print('Gradient used on Original Image')
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
show_pic(gradient)




