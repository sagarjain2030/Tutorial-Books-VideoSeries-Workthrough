# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 23:31:09 2018

@author: Sagar
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
img1 = cv2.imread('../Data/dog_backpack.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.imread('../Data/watermark_no_copy.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

#print('image 1 shape is  ' + str(img1.shape))
#plt.imshow(img1)

#print('image 2 shape is  ' + str(img2.shape))
#plt.imshow(img2)


### Blending Image of Same size
#print("Resizing images to (1200,1200,3)")

img1 = cv2.resize(img1,(1200,1200))
#plt.imshow(img1)

img2 = cv2.resize(img2,(1200,1200))
#plt.imshow(img2)

## Negative Gamma will make whole image darker
print('Blending Images ')
img_blend = cv2.addWeighted(src1 = img1,alpha = 0.85,src2 = img2, beta = 0.15,gamma = -10.0)
plt.imshow(img_blend)
'''
'''
### Overlay small image on top of large image. It means no blending
print("Overlaying Image one on another.Resizing image 2 to make it smaller image")
large_img = cv2.imread('../Data/dog_backpack.png')
large_img = cv2.cvtColor(large_img,cv2.COLOR_BGR2RGB)
small_img = cv2.imread('../Data/watermark_no_copy.png')
small_img = cv2.cvtColor(small_img,cv2.COLOR_BGR2RGB)
small_img = cv2.resize(small_img,(600,600))

x_offset = 0
y_offset = 0

x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]

large_img[y_offset:y_end,x_offset:x_end] = small_img
plt.imshow(large_img)
'''

### Masking for different size images

#print("Blending 2 different size images with masking")
img1 = cv2.imread('../Data/dog_backpack.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.imread('../Data/watermark_no_copy.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2,(600,600))

x_offset = 934 - 600
y_offset = 1401 - 600
row,cols,channels = img2.shape

roi = img1[y_offset:1401,x_offset:934]
#print('region of interest for blending images')
#plt.imshow(roi)

#print("Second image for masking")
img2gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
mask_inv = cv2.bitwise_not(img2gray)
#plt.imshow(mask_inv,cmap='gray')
#print(mask_inv.shape)

white_back = np.full(img2.shape,255,dtype=np.uint8)
#print('white background shape is ' + str(white_back.shape))

bk = cv2.bitwise_or(white_back,white_back,mask=mask_inv)
#plt.imshow(bk)

#print("Creating foreground mask ")
foreground = cv2.bitwise_or(img2,img2,mask=mask_inv)
#plt.imshow(foreground)

#print('final region of Interest for masking')
final_roi = cv2.bitwise_or(roi,foreground)
#print(final_roi.shape)
#plt.imshow(final_roi)

print("final_output image is ")
large_img= img1
small_img = final_roi

large_img[y_offset:y_offset+small_img.shape[0],x_offset:x_offset+small_img.shape[1]] = small_img
plt.imshow(large_img)
