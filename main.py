# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 22:02:12 2022

@author: 70115156
"""

#%% Imports
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from CNNetwork import CNNetwork, CNNetworkError
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

#%% Global variables
ListLabels=[]
Data=[] 

# %% 
# Data set location
ImgLocation="augmented_seg_20x20/"

# List image categories we are interested in
LABELS = set(["yes", "no"])
encoded_labels = list()
for label in LABELS:
    for img_name in list(os.listdir(ImgLocation+label)):
        #ImagePaths.append([ImgLocation+label+"/"+img_name])
        ListLabels.append([label])
        if label == 'yes':
            encoded_labels.append(1)
        elif label == 'no':
            encoded_labels.append(0)
        else:
            raise RuntimeError("ERR: Unknown label type detected")
        image = cv2.imread(ImgLocation+label+"/"+img_name,0) #gray scale conversion
        Data.append(image/255.0) #normalisation
        


# enc = OrdinalEncoder()
# encoded_labels=enc.fit_transform(np.array(ListLabels).reshape(-1,1))
Data = np.array(Data)

train_data, test_data, train_label, test_label = train_test_split(Data,encoded_labels, test_size=0.20, random_state=4, stratify=encoded_labels)


cnn = CNNetwork(Data[0])

# Training
for epoch in range (0, 2): #Epochs = 2
    no_of_correct = 0
    for i in range (0,train_data.shape[0]):
        output, output_loss, _is_correct = cnn.forward_prop(train_data[i], train_label[i])
        # print("\nNetwork output: {}, Target: {} ,Loss: {},total: {}".format(output, train_label[i], output_loss,output[0]+output[1]))
        
        no_of_correct += _is_correct
        cnn.back_propagate()
    Training_accuracy = no_of_correct/train_data.shape[0] * 100 # Accuracy calculation
    print("\n\n Training accuracy epoch {}: {:.2f}%".format(epoch,Training_accuracy))
    
pred = []

# Testing
no_of_correct = 0
for i in range (0,test_data.shape[0]):
    output, output_loss, _is_correct = cnn.forward_prop(test_data[i], test_label[i])
    print("\nNetwork output: {}, Target: {} ,Loss: {},".format(output, test_label[i], output_loss))
    
    no_of_correct += _is_correct
    pred.append(np.argmax(output))
   
    
# Accuracy calculation

# Testing_accuracy = no_of_correct/test_data.shape[0] * 100
# print("\n\n Testing accuracy: {:.2f}%".format(Testing_accuracy))
cm = confusion_matrix(test_label, np.array(pred))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['Negative','Positive']))
disp.plot()

plt.show()

print(cm)
print(classification_report(test_label, np.array(pred)))
print("CNN accuracy : ",accuracy_score(test_label,np.array(pred)))

# plot_confusion_matrix(cnn, test_data, test_label)  
# plt.show()