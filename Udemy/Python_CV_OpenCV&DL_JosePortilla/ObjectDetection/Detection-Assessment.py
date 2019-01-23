#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.pieriandata.com"><img src="../DATA/Logo.jpg"></a>
# *Copyright by Pierian Data Inc.*

# # Object Detection Assessment Project Exercise
# 
# ## Russian License Plate Blurring
# 
# Welcome to your object detection project! Your goal will be to use Haar Cascades to blur license plates detected in an image!
# 
# Russians are famous for having some of the most entertaining DashCam footage on the internet (I encourage you to Google Search "Russian DashCam"). Unfortunately a lot of the footage contains license plates, perhaps we could help out and create a license plat blurring tool?
# 
# OpenCV comes with a Russian license plate detector .xml file that we can use like we used the face detection files (unfortunately, it does not come with license detectors for other countries!)
# 
# ----
# 
# 
# #### 3 Ways to Approach this project:
# * Just go for it! Use the image under the DATA folder called car_plate.jpg and create a function that will blur the image of its license plate. Check out the Haar Cascades folder for the correct pre-trained .xml file to use.
# * Use this notebook! Here we offer a guide of what main steps you should take to complete the project.
# * Jump to the solutions notebook and video to treat this entire project as code-along project where you can code along with us.
# 
# ## Project Guide
# 
# Follow and complete the tasks below to finish the project!

# **TASK: Import the usual libraries you think you'll need.**

# In[10]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
get_ipython().run_line_magic('matplotlib', 'inline')


# **TASK: Read in the car_plate.jpg file from the DATA folder.**

# In[4]:


img = cv2.imread('../Data/car_plate.jpg')


# **TASK: Create a function that displays the image in a larger scale and correct coloring for matplotlib.**

# In[7]:


def display(img):
    # fill me in!
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    
    ax.imshow(img)


# In[8]:


display(img)


# In[4]:


display(img)


# **TASK: Load the haarcascade_russian_plate_number.xml file.**

# ### Since 2 haarcascade files are provided, 2 different function are used

# In[11]:


russian_plate_cascade_1 = cv2.CascadeClassifier('../Data/haarcascades/haarcascade_russian_plate_number.xml')
russian_plate_cascade_2 = cv2.CascadeClassifier('../Data/haarcascades/haarcascade_licence_plate_rus_16stages.xml')


# **TASK: Create a function that takes in an image and draws a rectangle around what it detects to be a license plate. Keep in mind we're just drawing a rectangle around it for now, later on we'll adjust this function to blur. You may want to play with the scaleFactor and minNeighbor numbers to get good results.**

# In[16]:


def detect_plate_1(img):
    img_russian_plate_1 = copy.deepcopy(img)
    plate_rect = russian_plate_cascade_1.detectMultiScale(img,1.2,5)
    for (x,y,w,h) in plate_rect:
        cv2.rectangle(img_russian_plate_1,(x,y),(x+w,y+h),(255,255,255),10)
    
    return img_russian_plate_1


# In[17]:


result = detect_plate_1(img)
display(result)


# In[ ]:


def detect_plate_2(img):
    img_russian_plate_2 = copy.deepcopy(img)
    plate_rect = russian_plate_cascade_2.detectMultiScale(img,1.1,2)
    for (x,y,w,h) in plate_rect:
        cv2.rectangle(img_russian_plate_2,(x,y),(x+w,y+h),(255,255,255),10)
    
    return img_russian_plate_2


# In[31]:


result = detect_plate_2(img)
display(result)


# ## Since first classifier is giving better result, function detect_plate_1 will be used

# **FINAL TASK: Edit the function so that is effectively blurs the detected plate, instead of just drawing a rectangle around it. Here are the steps you might want to take:**
# 
# 1. The hardest part is converting the (x,y,w,h) information into the dimension values you need to grab an ROI (somethign we covered in the lecture 01-Blending-and-Pasting-Images. It's simply [Numpy Slicing](https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python), you just need to convert the information about the top left corner of the rectangle and width and height, into indexing position values.
# 2. Once you've grabbed the ROI using the (x,y,w,h) values returned, you'll want to blur that ROI. You can use cv2.medianBlur for this.
# 3. Now that you have a blurred version of the ROI (the license plate) you will want to paste this blurred image back on to the original image at the same original location. Simply using Numpy indexing and slicing to reassign that area of the original image to the blurred roi.

# In[60]:


def detect_and_blur_plate(img):
    
    # fill me in
    img_russian_plate = img.copy()
    roi = img.copy()
    
    plate_rect = russian_plate_cascade_1.detectMultiScale(img,1.2,5)
    
    for (x,y,w,h) in plate_rect:
               
        roi = roi[y:y+h,x:x+w]
        blurred_roi = cv2.medianBlur(roi,15)
        img_russian_plate[y:y+h,x:x+w] = blurred_roi    
    return img_russian_plate
    


# In[61]:


result = detect_and_blur_plate(img)
display(result)


# # Great Job!
