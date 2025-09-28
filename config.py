import os 

RAW_DATA_DIR = "C:/Users/sadha/Documents/Projects/AI_Brain_tumor_detection/data/train_val"                 # raw images: data/raw/no/*.jpg, data/raw/yes/*.jpg
SEG_DIR      = "C:/Users/sadha/Documents/Projects/AI_Brain_tumor_detection/data/segmentation_results"  # segmentation outputs
AUG_DIR      = "C:/Users/sadha/Documents/Projects/AI_Brain_tumor_detection/data/augmented_results"     # augmentation outputs
RESIZED_DIR  = "C:/Users/sadha/Documents/Projects/AI_Brain_tumor_detection/data/resized_results"       # resized/padded outputs (not used by main script, but can be useful for inspection)
MODEL_PATH = os.getenv("MODEL_PATH", "model.npz")  # path to the trained model file
LABELS = ["yes", "no"]   # yes -> 1, no -> 0
EXPECTED_SIZE = 200  # images will be resized/padded to this size
EPOCHS = 2
RANDOM_STATE = 4
MODEL_PATH = "model.npz"