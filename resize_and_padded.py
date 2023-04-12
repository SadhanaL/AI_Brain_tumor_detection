# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:29:06 2022

@author: sadha
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2




Datadir = "Augmented_seg_200x200"
Result_dir = "Augmented_seg_20x20"
# The expected size variable controls the size of the resultant square matrix H*W
EXPECTED_SIZE = 20



Categories = ["no", "yes"]

def margins(new_size):
      
      right = int((EXPECTED_SIZE-new_size[0])/2) + (EXPECTED_SIZE-new_size[0])%2
      left = int((EXPECTED_SIZE-new_size[0])/2) 
      top = int((EXPECTED_SIZE-new_size[1])/2) + (EXPECTED_SIZE-new_size[1])%2
      bottom = int((EXPECTED_SIZE-new_size[1])/2) 
      return top, bottom, right, left


for category in Categories:
    path = os.path.join(Datadir, category)
    for img in os.listdir(path):
        print('\n\nImage name:'+str(img))
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        width = int(img_array.shape[1])
        height = int(img_array.shape[0])
        dimension =[width, height]
        maxi = np.max(dimension)
        for scale_factor in range(1,100):
            sf = scale_factor/100
            if((maxi*sf) >= EXPECTED_SIZE):
                sf= (scale_factor-1)/100
                break
        # print(sf)
     
        new_size = (int(width*sf),int(height*sf))
        print ('New Size: '+ str(new_size))
        resized = cv2.resize(img_array, new_size, interpolation=cv2.INTER_AREA)
        
        # cv2.imshow('original image',img )
        # cv2.imshow('resized',resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        



        if new_size[0] < EXPECTED_SIZE or new_size[1] < EXPECTED_SIZE:
            top, bottom, left, right = margins(new_size)
            print('Top: {}, Bottom: {}, Right: {}, Left: {}'.format(top, bottom, right, left))
            resized = cv2.copyMakeBorder(resized,top, bottom, left, right, cv2.BORDER_CONSTANT,None, value = 0)
            # cv2.imshow('padded_img', padded)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        print('Resized shape: ' + str(resized.shape))
        cv2.imwrite(Result_dir+'/'+category+'/'+img, resized)
        

       
     
          
        