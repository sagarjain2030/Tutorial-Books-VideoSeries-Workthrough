# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 21:55:03 2018

@author: Sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('../Data/00-puppy.jpg')

### No Error will be thrown if file is not read properly
#img = cv2.imread('00-puppy.jpg')

##checking type of img to make sure if imgae is read correctly or not
if(isinstance(img,np.ndarray)):
    #pass
    print(type(img))

print('shape of image' + str(img.shape))

print("Showing Originally read Image")
plt.imshow(img)

### MATPLOTLIB expects => RGB order channel
### OpenCV gives => BGR order channel
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("Converting Image in RGB format")
plt.imshow(fix_img)


### Reading Image in Gray scale
img_gray = cv2.imread('../Data/00-puppy.jpg',cv2.IMREAD_GRAYSCALE)
print('shape of image' + str(img_gray.shape))
print("Reading Image in Gray scale")
plt.imshow(img_gray,cmap='gray')

###Resizing Image
print("Resizing Image width > Height")
resize_img_1 = cv2.resize(fix_img,(1000,400))
plt.imshow(resize_img_1)

print("Resizing Image width < Height")
resize_img_2 = cv2.resize(fix_img,(400,1000))
plt.imshow(resize_img_2)

# Resizing image by ratio
widthRatio = 0.5
HeightRatio = 0.75
print("Resizing Image widthRatio < HeightRatio")
resize_img_3 = cv2.resize(fix_img,(0,0),fix_img,widthRatio,HeightRatio)
print('shape of image' + str(resize_img_3.shape))
plt.imshow(resize_img_3)

widthRatio = 0.75
HeightRatio = 0.5
print("Resizing Image widthRatio > HeightRatio")
resize_img_4 = cv2.resize(fix_img,(0,0),fix_img,widthRatio,HeightRatio)
print('shape of Image' + str(resize_img_4.shape))
plt.imshow(resize_img_4)

### Flipping Image
print("Flipping Image horizontally")
flip_image_horizontal = cv2.flip(fix_img,0)
plt.imshow(flip_image_horizontal)

print("Flipping Image vertically")
flip_image_vertical = cv2.flip(fix_img,1)
plt.imshow(flip_image_vertical)

print("Flipping Image in both direction")
flip_image_both = cv2.flip(fix_img,-1)
plt.imshow(flip_image_both)

### Writing New File and showing it using cv2 function
cv2.imwrite('../Data/NewImg.png',fix_img)
image = cv2.imread('../Data/NewImg.png')

while True:
    cv2.imshow('NewImage',image)
    if cv2.waitKey(10) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()
