# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 10:53:28 2018

@author: Sagar
"""
import cv2
import matplotlib.pyplot as plt

def show_pic(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    

img_original = cv2.imread('../data/rainbow.jpg')
print('Original Image')
show_pic(img_original)

##### Reading Image in gray scale directly. Remember to mention cmap= Gray
img = cv2.imread('../data/rainbow.jpg',0)
print('Image in Grayscale')
show_pic(img)

print('Binary Thresholding with 127 as threshold value')
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
show_pic(thresh1)

print('Inverse Binary Thresholding with 127 as threshold value')
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
show_pic(thresh1)

print('Threshold Truncation with 127 as threshold value and 255 as replacement')
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
show_pic(thresh1)

print('Threshold Truncation with 127 as threshold value and 0 as replacement')
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
show_pic(thresh1)

print('Inverse Threshold Truncation with 127 as threshold value and 255 as replacement')
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
show_pic(thresh1)

### Adaptive Thresholding
print("Adaptive Thresholding")
img = cv2.imread('../data/crossword.jpg',0)
show_pic(img)

ret,thresh1 = cv2.threshold(img,180,255,cv2.THRESH_BINARY)
show_pic(thresh1)

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
show_pic(th2)

print("Blending Binary and adaptive")
blended = cv2.addWeighted(src1=thresh1,alpha=0.6,src2=th2,beta=0.4,gamma=0)
show_pic(blended)