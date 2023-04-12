# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:13:39 2022

@author: sadha
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

#%%

# # cv2.imshow('image', img)
# # img_array = np.asarray(img)
# # print (img_array)
# print(img.shape)
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray= cv2.GaussianBlur(img,(5,5),0)
# # hpf = img - cv2.GaussianBlur(gray, (21, 21), 3)+127
# median = cv2.medianBlur(gray, 3)
# sobel = cv2.Sobel(median, cv2.CV_64F, 1, 1)
# kernel = np.ones((5, 5), np.uint8)
# img_erosion = cv2.erode(sobel, kernel, iterations=1)
# img_dialation = cv2.dilate(img, kernel, iterations=1)
# # plt.hist(img_dialation)
# thresh= cv2.adaptiveThreshold(img_dialation,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY,21,5)
#%%
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

def show_results(img, res, label):
    plt.Figure()
    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(2,2,2)
    plt.imshow(res,cmap='gray')
    plt.title('Result Image - '+str(label))
    # plt.subplot(2,2,3)
    # plt.hist(img)
    plt.show()
    return True

#%%
def grab_cut(img, res, label):
    blur = cv.GaussianBlur(res,(31,31), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(blur)
    print(maxLoc)
    print("Max location: {},{}".format(maxLoc[0],maxLoc[1]))
    rect = (max(0,maxLoc[0]-60),
            max(0,maxLoc[1]-60),
            60,60)
    
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    seg = img*mask2[:,:,np.newaxis]
    
    # ## With mask
    # # newmask is the mask image I manually labelled
    # mask = np.zeros(img.shape[:2],np.uint8)
    # newmask = res
    # bgdModel = np.zeros((1,65),np.float64)
    # fgdModel = np.zeros((1,65),np.float64)
    # # wherever it is marked white (sure foreground), change mask=1
    # # wherever it is marked black (sure background), change mask=0
    # mask[newmask == 0] = 0
    # mask[newmask == 255] = 1
    # mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
    # mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    # mask_seg = img*mask[:,:,np.newaxis]
    
    plt.Figure()
    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image - Expected result : {} '.format(label))
    plt.subplot(2,2,2)
    plt.imshow(res,cmap='gray')
    plt.title("Max location: {},{}".format(maxLoc[0],maxLoc[1]))
    plt.subplot(2,2,3)
    plt.imshow(seg,cmap='gray')
    # plt.subplot(2,2,4)
    # plt.imshow(res,cmap='gray')
    # plt.imshow(mask_seg)
    plt.show()


# mask[mask == 30] = 0
# mask[mask == 255] = 1
# mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
# mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask[:,:,np.newaxis]
# plt.imshow(img),plt.colorbar(),plt.show()

#%%

if __name__ == '__main__':
    Image_Dataset = ('dataset_20x20/')
    Result_dir = "Segmentation_20x20"
    LABELS = set(["yes", "no"])
    encoded_labels = list()
    for label in LABELS:
        for img_name in list(os.listdir(Image_Dataset+label)):
            img = cv.imread(Image_Dataset+label+"/"+img_name)
            res = image_segmentation(img, label)
            #show_results(img, res, label)
            #grab_cut(img, res, label)
            cv.imwrite(Result_dir+'/'+str(label)+'/'+img_name, res)


