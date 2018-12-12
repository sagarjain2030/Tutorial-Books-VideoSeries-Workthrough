# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 22:08:37 2018

@author: Sagar
"""

#### PIL (Pillow) Python Imaging Library: Required to open Jpeg or png image 

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

pic = Image.open("../Data/00-puppy.jpg")
#pic.show()

#print(type(pic))

pic_arr = np.asarray(pic)
#print(type(pic_arr))

#print(pic_arr.shape)
#plt.imshow(pic_arr)

pic_red = pic_arr.copy()
pic_red = pic_red[:,:,0]
#plt.imshow(pic_red)

pic_green = pic_arr.copy()
pic_green = pic_green[:,:,1]
#plt.imshow(pic_green)

pic_blue = pic_arr.copy()
pic_blue = pic_blue[:,:,2]
#plt.imshow(pic_blue)

pic_NoGreen = pic_arr.copy()
pic_NoGreen[:,:,1] = 0
#plt.imshow(pic_NoGreen,cmap='gray')

pic_NoBlue = pic_arr.copy()
pic_NoBlue[:,:,2] = 0
#plt.imshow(pic_NoBlue,cmap='gray')

pic_NoRed = pic_arr.copy()
pic_NoRed[:,:,0] = 0
#plt.imshow(pic_NoRed,cmap='gray')

pic_NoBlue[:,:,0] = 0
#plt.imshow(pic_NoBlue,cmap='gray')
plt.imshow(pic_NoBlue)
