# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:53:08 2022

@author: sadha
"""
import os
import cv2
from PIL import Image

Datadir = "Segmentation_200x200/"
Result_dir = "Augmented_seg_200x200"

Categories = ["no", "yes"]
count = 1

for category in Categories:
    path = os.path.join(Datadir, category)
    for img_name in os.listdir(path):
        img = Image.open(Datadir+category+'/'+img_name)
        hor_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        ver_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
        deg_flip = img.transpose(Image.ROTATE_90)
        rotated45 = img.rotate(45)
        rotated135 = img.rotate(135)
        rotated225 = img.rotate(225)
        rotated315 = img.rotate(315)
        # img.show()
        # hor_flip.show()
        # ver_flip.show()
        # deg_flip.show()
        # rotated.show()
        if category == 'yes':
            prefix = 'Y'
        else:
            prefix = 'N'
        img.save(Result_dir+'/'+category+'/'+prefix+str(count)+'.jpg')
        count+=1
        hor_flip.save(Result_dir+'/'+category+'/'+prefix+str(count)+'.jpg')
        count+=1
        ver_flip.save(Result_dir+'/'+category+'/'+prefix+str(count)+'.jpg')
        count+=1
        deg_flip.save(Result_dir+'/'+category+'/'+prefix+str(count)+'.jpg')
        count+=1
        rotated45.save(Result_dir+'/'+category+'/'+prefix+str(count)+'.jpg')
        count+=1
        rotated135.save(Result_dir+'/'+category+'/'+prefix+str(count)+'.jpg')
        count+=1
        rotated225.save(Result_dir+'/'+category+'/'+prefix+str(count)+'.jpg')
        count+=1
        rotated315.save(Result_dir+'/'+category+'/'+prefix+str(count)+'.jpg')
        count+=1
print('Augmentation Done')