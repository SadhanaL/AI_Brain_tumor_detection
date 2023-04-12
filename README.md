# AI_Brain_tumor_detection
Brain tumor segmentation and detection from MRI images using CNN and KNN algorithms implemented from scratch

---------------
The dataset was obtained from kaggle computer vision with a limited MRIs of 155 with tumor and 98 without tumor images. The dataset can be found here: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/code

Augmented_PIL_trial - Performs augmentation of the dataset using flip and rotate in the pillow package

processed_img(1) - Segmentation of tumor (ROI)

resize_and_padded - Images are resized to 20x20 and padded if required

CNNetwork - Implemetation of CNN from scratch

main - main of CNN

feature extraction - Using GLCM extracting features from the segemented images

KNN - Implements KNN from scratch using the csv file where the features are stored from the previous feature extraction file.
