# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 19:50:40 2019

@author: Sagar
"""

import cv2
import numpy as np 
from matplotlib import pyplot as plt

def show_pic(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    
def load_img():
    img = cv2.imread('../Data/bricks.jpg').astype(np.float32)/255
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

print("original Image")

img = load_img()
show_pic(img)

#### Gamma value changing
print("Next 4 images: Gamma values 1/4,1/10,2,8")
gamma = 1/4
result = np.power(img,gamma)
show_pic(result)

gamma = 1/10
result = np.power(img,gamma)
show_pic(result)

gamma = 2
result = np.power(img,gamma)
show_pic(result)

gamma = 8
result = np.power(img,gamma)
show_pic(result)

#### Blurring of Image
print("Next 7 images : first original with written brick on it. Others different kernel sizes and kernel internal values")
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text = "bricks",org=(10,600),fontFace=font,fontScale=10,color=(255,0,0),thickness=4)
show_pic(img)

kernel = np.ones(shape=(5,5),dtype=np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
show_pic(dst)

#### As kernel size increases image gets lighter while as kernel values decrease image gets darker
kernel = np.ones(shape=(3,3),dtype=np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
show_pic(dst)

kernel = np.ones(shape=(9,9),dtype=np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
show_pic(dst)

kernel = np.ones(shape=(9,9),dtype=np.float32)/125
dst = cv2.filter2D(img,-1,kernel)
show_pic(dst)

kernel = np.ones(shape=(11,11),dtype=np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
show_pic(dst)

kernel = np.ones(shape=(9,9),dtype=np.float32)/325
dst = cv2.filter2D(img,-1,kernel)
show_pic(dst)

#### Default blurring in OpenCV
print("Next 2 images, default kernel function in CV2")
blurred = cv2.blur(img,ksize=(5,5))
show_pic(blurred)

blurred = cv2.blur(img,ksize=(10,10))
show_pic(blurred)

print("Next image, Gaussian Blur in CV2")
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text = "bricks",org=(10,600),fontFace=font,fontScale=10,color=(255,0,0),thickness=4)

gaussianBlur_1 = cv2.GaussianBlur(img,(5,5),10)
show_pic(gaussianBlur_1)

print("Next image, Median Blur in CV2")
medianBlur_1 = cv2.medianBlur(img,5)
show_pic(medianBlur_1)

print("Next images noisy dog images and median blur applied to it")
img = cv2.imread('../Data/sammy.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
show_pic(img)

img = cv2.imread('../Data/sammy_noise.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
show_pic(img)

median = cv2.medianBlur(img,5)
show_pic(median)

print("Last image is bilateral filtering")
#### Bilateral Blurring:
img = load_img()
bilateralBlur = cv2.bilateralFilter(img,9,75,75)
show_pic(bilateralBlur)


