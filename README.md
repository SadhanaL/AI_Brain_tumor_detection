# AI_Brain_tumor_detection
Brain tumor segmentation and detection from MRI images using CNN algorithm implemented from scratch

---------------
The dataset was obtained from kaggle computer vision with a limited MRIs of 155 with tumor and 98 without tumor images. The dataset can be found here: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/code

utils.py - Performs Segmentation of tumor (ROI), augmentation of the dataset using flip and rotate in the pillow package and resizes the images to the EXPECTED_SIZE given in the config.py

CNNetwork.py - Implemetation of CNN from scratch (Conv, max pool, linear, softmax)

main.py - Main of CNN where the raw data is loaded, preprocessed, trained and validated

app.py - Defines a FastAPI application with a /predict endpoint that accepts an uploaded image, runs model inference, and returns the classification result (tumor or not tumor).

# 🧠 Tumor Detection API

This project provides a FastAPI application that classifies images as **tumor** or **no tumor** using a trained CNN model.

---

## 🏋️‍♂️ Training the Model

1. **Prepare the data**  
   Place your raw training images inside the folder: data/train_val

2. **Run training**  
From the project root, execute:

```bash
python main.py
```

This will train the model and save it to model.npz (or whatever path is configured in config.py.

3. **Run the API**  
Install dependencies
```bash
pip install -r requirements.txt
```
4. **Start the FastAPI server** 
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
5. **Go to Swagger UI**
```bash
http://127.0.0.1:8000/docs
```
Upload a test image and get a classification :) 