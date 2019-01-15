# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:28:46 2019

@author: Sagar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_pic(img,cmap='gray'):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)

reeses = cv2.imread('../Data/reeses_puffs.png',0)
show_pic(reeses)

cereals = cv2.imread('../Data/many_cereals.jpg',0)
show_pic(cereals)

#### Brute Force Detection with ORB Detectors
orb = cv2.ORB_create()
kep1 , desc1 = orb.detectAndCompute(reeses,None)
kep2 , desc2 = orb.detectAndCompute(cereals,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches = bf.match(desc1,desc2)
matches = sorted(matches,key=lambda x:x.distance)
reeses_matches  = cv2.drawMatches(reeses,kep1,cereals,kep2,matches[:25],None,flags=2)
show_pic(reeses_matches)


#### SIFT : Scale Invariant Feature Transform
sift = cv2.xfeatures2d.SIFT_create()
kep1, desc1 = sift.detectAndCompute(reeses,None)
kep2, desc2 = sift.detectAndCompute(cereals,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1,desc2,k=2)

### Ratio Test
### Less Distance == Better Match
good = []
for match1,match2 in matches:
    if match1.distance < 0.75*match2.distance:
        good.append([match1])
        
sift_matches = cv2.drawMatchesKnn(reeses,kep1,cereals,kep2,good,None,flags=2)
show_pic(sift_matches)

#### FLANN based Matches instead of Brute Force
sift = cv2.xfeatures2d.SIFT_create()
kep1, desc1 = sift.detectAndCompute(reeses,None)
kep2, desc2 = sift.detectAndCompute(cereals,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,tree=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(desc1,desc2,k=2)

good = []
for match1,match2 in matches:
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

flann_matches = cv2.drawMatchesKnn(reeses,kep1,cereals,kep2,good,None,flags=2)
show_pic(flann_matches)

matchesMask = [[0,0] for i in range(len(matches))]
good = []
for i,(match1,match2) in enumerate(matches):
    if match1.distance < 0.75*match2.distance:
        matchesMask[i] = [1,0]
        
draw_paramas = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),
                    matchesMask=matchesMask,flags=0)
flann_matches = cv2.drawMatchesKnn(reeses,kep1,cereals,kep2,matches,None,**draw_paramas)
show_pic(flann_matches)








