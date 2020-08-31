# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:03:06 2020

@author: DHRUV
"""


import cv2
import os
import numpy as np
import faceRecognition as frg
test_img = cv2.imread('C:/Users/DHRUV/Desktop/ML projects/Face Recognition System/test_images/IMG_6423.jpg')
faces_detect, gray_img = frg.faceDetect(test_img)
print(faces_detect)

for (x, y, w, h) in faces_detect:
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness = 5)
    
resize_img = cv2.resize(test_img, (1000, 700))
cv2.imshow("face detect", resize_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
