# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:41:50 2020

@author: DHRUV
"""


import cv2
import os
import numpy as np

# function for face detection to detect the faces from laptop camera as of now
def faceDetect(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('C:/Users/DHRUV/Desktop/ML projects/Face Recognition System/haar_cascade/haarcascade_frontalface_default.xml') 
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.10, minNeighbors = 6)
    return faces, gray_img