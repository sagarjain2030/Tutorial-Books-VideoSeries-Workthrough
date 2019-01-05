# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 17:27:38 2019

@author: Sagar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

def display_img(img,cmap=False):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    if cmap:
        ax.imshow(img,cmap='gray')
    else:
        ax.imshow(img)
    
def show_Hist(hist):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.plot(hist)

print("Image Listing")
print('1 ---->  Horse Image\n2 ---->  Rainbow Image\n3 ---->  Bricks Image\n\
4 ---->  Blue Channel of Brick Image\n5 ---->  Blue Channel of Horse Image\n\
6 ---->  All channels for Blue Brick Image\n7 ---->  All channels for Rainbow Image\n\
7 ---->  All channels for Horse Image\n8 ---->  Mask Image\n\
9 ---->  masked Rainbow Image\n10 ----> histogram of red channel for masked Rainbow Image\n\
11 ----> histogram of red channel for Nonmasked Rainbow Image\n\
12 ----> Gorila Image in grayscale\n13 ----> Gorila Histogram Image in grayscale\n\
14 ----> Equalised Gorila Image in grayscale\n15 ----> Equalised Gorila Histogram Image in grayscale\n\
16 ----> Gorila Image in color\n17 ----> Equalised Gorila Image in Color')

dark_horse = cv2.imread('../data/horse.jpg')
show_horse = cv2.cvtColor(dark_horse,cv2.COLOR_BGR2RGB)
display_img(show_horse)

rainbow = cv2.imread('../data/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow,cv2.COLOR_BGR2RGB)
display_img(show_rainbow)

blue_bricks = cv2.imread('../data/bricks.jpg')
show_blue_bricks = cv2.cvtColor(blue_bricks,cv2.COLOR_BGR2RGB)
display_img(show_blue_bricks)

### For OpenCV color channel order is BGR so channel 0 is Blue and Channel 2 is Red
hist_values = cv2.calcHist([blue_bricks],channels=[0],mask=None,histSize=[256],ranges=[0,256])          
show_Hist(hist_values)

### Blue channel for Dark Horse
hist_horse_blue = cv2.calcHist([dark_horse],channels=[0],mask=None,histSize=[256],ranges=[0,256])          
show_Hist(hist_horse_blue)

### # color Histogram for blue brick Image
img = blue_bricks
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    ax.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Blue Bricks Image')
plt.show()

#### color Histogram for Rainbow Image
img = rainbow
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    ax.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Rainbow Image')
plt.show()

#### color Histogram for Horse Image
img = dark_horse
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    ax.plot(histr,color = col)
    plt.xlim([0,50])
    plt.ylim([0,100000])
plt.title('Horse Image')
plt.show()


###  Histogram Masking
img = copy.deepcopy(rainbow)
print(img.shape)

mask = np.zeros(img.shape[:2],np.uint8)
mask[300:400,100:400] = 255
display_img(mask,cmap=True)

### First img performs operation.Useful for further image processing.Second will be for visualization purpose
masked_img = cv2.bitwise_and(img,img,mask=mask)
show_masked_img = cv2.bitwise_and(show_rainbow,show_rainbow,mask=mask)
display_img(show_masked_img)

hist_mask_red_rainow = cv2.calcHist([rainbow],[2],mask,[256],[0,256])
show_Hist(hist_mask_red_rainow)

hist_red_rainow = cv2.calcHist([rainbow],[2],None,[256],[0,256])
show_Hist(hist_red_rainow)

###  Histogram Equilization
gorilla = cv2.imread('../data/gorilla.jpg',0)
display_img(gorilla,cmap=True)

hist_gorilla = cv2.calcHist([gorilla],[0],None,[256],[0,256])
show_Hist(hist_gorilla)

eq_gorilla = cv2.equalizeHist(gorilla)
display_img(eq_gorilla,True)

eq_hist_gorilla = cv2.calcHist([eq_gorilla],[0],None,[256],[0,256])
show_Hist(eq_hist_gorilla)

color_gorilla = cv2.imread('../data/gorilla.jpg')
show_color_gorilla = cv2.cvtColor(color_gorilla,cv2.COLOR_BGR2RGB)
display_img(show_color_gorilla)

#### To apply equalised histogram function , it is necessary to convert color 
#### RGB image to HSV image. Now equalization will only be done for Value
#### channel of HSV
hsv = cv2.cvtColor(color_gorilla,cv2.COLOR_BGR2HSV)
hsv_eq = copy.deepcopy(hsv)
hsv_eq[:,:,2] = cv2.equalizeHist(hsv_eq[:,:,2])
eq_color_gorilla = cv2.cvtColor(hsv_eq,cv2.COLOR_HSV2RGB)
display_img(eq_color_gorilla)


