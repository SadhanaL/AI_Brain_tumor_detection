import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from CNNetwork import CNNetwork
from config import *  # expects EPOCHS, RANDOM_STATE, MODEL_PATH, RAW_DATA_DIR, SEG_DIR, AUG_DIR
from utils import ensure_dir, run_segmentation, run_augmentation, resize_and_pad_images 


def load_data(img_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load grayscale images and labels from a directory structure.

    Parameters
    ----------
    img_dir : str
        Path to the directory containing subfolders for each label.

    Returns
    -------
    tuple
        X : np.ndarray
            Array of images of shape (N, H, W), where N is the number of images.
        y : np.ndarray
            Array of integer labels (0 for 'no', 1 for 'yes').

    Raises
    ------
    RuntimeError
        If images are missing, have inconsistent shapes, or loading fails.
    """
    try:
        X, y = [], []
        ref_shape = None

        for label in LABELS:
            label_dir = os.path.join(img_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for img_name in os.listdir(label_dir):
                p = os.path.join(label_dir, img_name)
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # sanity: ensure all images have same shape (pipeline should guarantee this)
                if ref_shape is None:
                    ref_shape = img.shape
                elif img.shape != ref_shape:
                    raise RuntimeError(
                        f"Inconsistent image shape found: {p} has {img.shape}, "
                        f"expected {ref_shape}. Check resize_and_pad step."
                    )

                img = img.astype(np.float32) / 255.0
                X.append(img)
                y.append(1 if label == "yes" else 0)

        if not X:
            raise RuntimeError(f"No images found in {img_dir}. Error in the preprocessing pipeline.")

        X = np.stack(X, axis=0).astype(np.float32)
        y = np.array(y, dtype=np.int64)
        return X, y
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}") from e

def main():
    """
    Main pipeline for preparing data, training, evaluating, and saving the CNNetwork model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Prepare directories
    for d in (RAW_DATA_DIR, SEG_DIR, AUG_DIR):
        ensure_dir(d)
        for label in LABELS:
            ensure_dir(os.path.join(d, label))

    # Segmentation -> SEG_DIR
    print(f"[1/3] Segmenting images from {RAW_DATA_DIR} -> {SEG_DIR}")
    run_segmentation(RAW_DATA_DIR, SEG_DIR)

    # Augmentation -> AUG_DIR
    print(f"[2/3] Augmenting segmented images from {SEG_DIR} -> {AUG_DIR}")
    run_augmentation(SEG_DIR, AUG_DIR)

    # Resize & Pad
    print(f"[3/4] Resizing & padding {AUG_DIR} -> {RESIZED_DIR}")
    resize_and_pad_images(AUG_DIR, RESIZED_DIR)

    # Load, train, evaluate
    print(f"[4/4] Loading data from {RESIZED_DIR}")
    X, y = load_data(RESIZED_DIR)

    H, W = X[0].shape
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    net = CNNetwork((H, W))
    for epoch in range(EPOCHS):
        correct = 0
        for xi, yi in zip(X_train, y_train):
            _, _, is_correct, grad_out = net.forward_prop(xi, yi)
            correct += is_correct
            net.back_propagate(grad_out)
        acc = 100.0 * correct / len(X_train)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {acc:.2f}%")

    # validation
    preds = []
    correct = 0
    for xi, yi in zip(X_test, y_test):
        p = net.forward_prop(xi)
        pred = int(np.argmax(p))
        preds.append(pred)
        correct += int(pred == yi)
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["no", "yes"]))
    print("Accuracy:", accuracy_score(y_test, preds))

    # save for API
    net.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
