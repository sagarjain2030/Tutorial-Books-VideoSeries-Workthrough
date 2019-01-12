# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:58:27 2019

@author: Sagar
"""

import cv2
import numpy as np
import  matplotlib.pyplot as plt

def show_pic(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)
    
   
flat_chessBoard = cv2.imread('../Data/flat_chessboard.png')
show_pic(flat_chessBoard)

#### Method specifically search for checker board or chess board pattern
#### Not a kind of but exactly the same
found,corners = cv2.findChessboardCorners(flat_chessBoard,(7,7))
if(found):
    cv2.drawChessboardCorners(flat_chessBoard,(7,7),corners,found)
    show_pic(flat_chessBoard)
    
dot_grid = cv2.imread('../Data/dot_grid.png')
show_pic(dot_grid)

found,corners = cv2.findCirclesGrid(dot_grid,(10,10),cv2.CALIB_CB_SYMMETRIC_GRID)
if(found):
    cv2.drawChessboardCorners(dot_grid,(10,10),corners,found)
    show_pic(dot_grid)
   
if(found):
    cv2.drawChessboardCorners(flat_chessBoard,(10,10),corners,found)
    show_pic(flat_chessBoard)