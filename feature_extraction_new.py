# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:57:22 2022

@author: sadha
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os 
from PIL import Image
from skimage.feature.texture import graycomatrix , graycoprops
import csv

        
def image_segmentation(img, label):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _threshold_val = gray.mean() +90
    
    # Thresholding
    ret, thresh = cv.threshold(gray,_threshold_val,255,cv.THRESH_BINARY)
    #thresh= cv.adaptiveThreshold(thresh,255,cv.ADAPTIVE_THRESH_MEAN_C ,cv.THRESH_BINARY,21,10)
    #ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 2))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # plt.imshow(closed) 
    
    blur = cv.GaussianBlur(closed,(3,3), 0)
    
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv.erode(blur, kernel, iterations=2)
    img_dilation = cv.dilate(img_erosion, kernel, iterations=5)
    
    # plt.Figure()
    # plt.subplot(3,3,1)
    # plt.imshow(gray, cmap='gray')
    # plt.title('Original Image - Mean: {}, Thresholds: {}'.format(str(gray.mean()),str(_threshold_val)))
    # plt.subplot(3,3,4)
    # plt.imshow(thresh,cmap='gray')
    # plt.title('Adaptive Thresh Image -'+str(label))
    # plt.subplot(3,3,5)
    # plt.imshow(blur,cmap='gray')
    # plt.title('Guassian Blur Image')
    # plt.subplot(3,3,6)
    # plt.imshow(img_erosion,cmap='gray')
    # plt.title('Erosion Image')
    # plt.subplot(3,3,7)
    # plt.imshow(img_dilation,cmap='gray')
    # plt.title('Dilation Image -'+str(label))
    # plt.show()
    
    result = img_dilation
    #show_results(gray, thresh, label)
    return result

def textureFeatures(img, label):
    GLCM = graycomatrix(img, [50], [0])
    energy = graycoprops(GLCM, 'energy')[0,0]
    corr = graycoprops(GLCM, 'correlation')[0, 0]
    homogen = graycoprops(GLCM, 'homogeneity')[0, 0]
    contrast = graycoprops(GLCM, 'contrast')[0, 0]
    
    GLCM2 = graycomatrix(img, [30], [0])
    energy2 = graycoprops(GLCM2, 'energy')[0,0]
    corr2 = graycoprops(GLCM2, 'correlation')[0, 0]
    homogen2 = graycoprops(GLCM2, 'homogeneity')[0, 0]
    contrast2 = graycoprops(GLCM2, 'contrast')[0, 0]
    
    GLCM3 = graycomatrix(img, [5], [0])
    energy3 = graycoprops(GLCM3, 'energy')[0,0]
    corr3 = graycoprops(GLCM3, 'correlation')[0, 0]
    homogen3 = graycoprops(GLCM3, 'homogeneity')[0, 0]
    contrast3 = graycoprops(GLCM3, 'contrast')[0, 0]
    
    GLCM4 = graycomatrix(img, [50], [np.pi/4])
    energy4 = graycoprops(GLCM4, 'energy')[0,0]
    corr4 = graycoprops(GLCM4, 'correlation')[0, 0]
    homogen4 = graycoprops(GLCM4, 'homogeneity')[0, 0]
    contrast4 = graycoprops(GLCM4, 'contrast')[0, 0]
    
    GLCM5 = graycomatrix(img, [30], [np.pi/2])
    energy5 = graycoprops(GLCM5, 'energy')[0,0]
    corr5 = graycoprops(GLCM5, 'correlation')[0, 0]
    homogen5 = graycoprops(GLCM5, 'homogeneity')[0, 0]
    contrast5 = graycoprops(GLCM5, 'contrast')[0, 0]
    return energy, corr, homogen, contrast, energy2, corr2, homogen2, contrast2,energy3, corr3, homogen3, contrast3, energy4, corr4, homogen4, contrast4, energy5, corr5, homogen5, contrast5


Image_Dataset = ('Augmented_seg_200x200/')
LABELS = set(["yes", "no"])
encoded_labels = list()


header = ['e1', 'cor1', 'h1', 'con1','e2', 'cor2', 'h2', 'con2', 'e3', 'cor3', 'h3', 'con3', 'e4', 'cor4', 'h4', 'con4', 'e5', 'cor5', 'h5', 'con5', 'label']
with open('new_feature.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for label in LABELS:    
        for img_name in list(os.listdir(Image_Dataset+label)):
            img = cv.imread(Image_Dataset+label+"/"+img_name)
            result = image_segmentation(img, label)
            en, corr, hm, cn, en2, corr2, hm2, cn2, en3, corr3, hm3, cn3, en4, corr4, hm4, cn4, en5, corr5, hm5, cn5 = textureFeatures(result, label)
            writer.writerow([str(en), str(corr), str(hm), str(cn), str(en2), str(corr2), str(hm2), str(cn2), str(en3), str(corr3), str(hm3), str(cn3), str(en4), str(corr4), str(hm4), str(cn4), str(en5), str(corr5), str(hm5), str(cn5), str(label)])