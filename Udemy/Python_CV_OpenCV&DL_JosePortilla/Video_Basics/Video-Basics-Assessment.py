#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.pieriandata.com"><img src="../DATA/Logo.jpg"></a>
# 
# <em text-align:center>Copyright Pierian Data Inc.</em>

# # Video Basics Assessment 
# 
# * **Note: This assessment is quite hard! Feel free to treat it as a code along and jump to the solutions** *
# 
# ## Project Task
# 
# **You only have one task here. Create a program that reads in a live stream from a camera on your computer (or if you don't have a camera, just open up a video file). Then whenever you click the left mouse button down, create a blue circle around where you've clicked. Check out the video for an example of what the final project should look like**

# **Guide**
# 
# * Create a draw_circle function for the callback function
# * Use two events cv2.EVENT_LBUTTONDOWN and cv2.EVENT_LBUTTONUP
# * Use a boolean variable to keep track if the mouse has been clicked up and down based on the events above
# * Use a tuple to keep track of the x and y where the mouse was clicked.
# * You should be able to then draw a circle on the frame based on the x,y coordinates from the Event 
# 
# Check out the skeleton guide below:

# In[5]:


import cv2


# In[6]:


# Create a function based on a CV2 Event (Left button click)

# mouse callback function
def draw_circle(event,x,y,flags,param):

    global center,clicked

    # get mouse click on down and track center
    if(event == cv2.EVENT_LBUTTONDOWN):
        center = (x,y)
        clicked = False
    if(event == cv2.EVENT_LBUTTONUP):
        clicked = True
        
    # Use boolean variable to track if the mouse has been released
        
# Haven't drawn anything yet!
center = (0,0)
clicked = False


# Capture Video
cap = cv2.VideoCapture(0)

# Create a named window for connections
cv2.namedWindow('LiveFeed')

# Bind draw_rectangle function to mouse cliks
cv2.setMouseCallback('LiveFeed',draw_circle)


while True:
    # Capture frame-by-frame
    ret,frame = cap.read()
    
    # Use if statement to see if clicked is true
    if clicked == True:
        # Draw circle on frame
        cv2.circle(frame,center=center,radius=50,color=(255,0,0),thickness=10)
        
        
    # Display the resulting frame
    cv2.imshow('LiveFeed',frame)

    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
   
### When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

