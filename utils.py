import os 
import cv2
import numpy as np
from typing import Optional, Union
from PIL import Image
from config import LABELS, EXPECTED_SIZE

# to create a new directory if it doesn't exist
def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str
        Path to the directory to create.

    Returns
    -------
    None
    """
    os.makedirs(path, exist_ok=True)

def image_segmentation(img_bgr: np.ndarray) -> np.ndarray:
    """
    Segment the tumor region from a BGR image using thresholding and morphological operations.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format.

    Returns
    -------
    np.ndarray
        Segmented binary image (single channel).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh_val = float(gray.mean()) + 90.0

    # Threshold
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Close gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # Blur + erode + dilate
    blur = cv2.GaussianBlur(closed,(3,3), 0)
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(blur, kernel, iterations=2)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=5)

    return img_dilation  

def run_augmentation(src_dir: str, dst_dir: str) -> None:
    """
    Perform data augmentation on images in a directory and save augmented images.

    Parameters
    ----------
    src_dir : str
        Source directory containing subfolders for each label.
    dst_dir : str
        Destination directory to save augmented images.

    Returns
    -------
    None
    """
    try:
        for label in LABELS:
            in_dir = os.path.join(src_dir, label)
            out_dir = os.path.join(dst_dir, label)
            ensure_dir(out_dir)
            if not os.path.isdir(in_dir):
                continue

            # maintain separate counters per class
            count = 1
            prefix = "Y" if label == "yes" else "N"

            for fname in os.listdir(in_dir):
                src_path = os.path.join(in_dir, fname)
                img = Image.open(src_path)

                # base + 7 augmentations = 8 images total (same as your script)
                variants = [
                    img,
                    img.transpose(Image.FLIP_LEFT_RIGHT),
                    img.transpose(Image.FLIP_TOP_BOTTOM),
                    img.transpose(Image.ROTATE_90),
                    img.rotate(45),
                    img.rotate(135),
                    img.rotate(225),
                    img.rotate(315),
                ]

                for im in variants:
                    out_name = f"{prefix}{count}.jpg"
                    out_path = os.path.join(out_dir, out_name)
                    im.save(out_path, format="JPEG")
                    count += 1
    except Exception as e:
        raise RuntimeError(f"Error during augmentation: {e}") from e

def compute_margins(new_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """
    Compute the padding margins needed to center an image of new_size in a square of EXPECTED_SIZE.

    Parameters
    ----------
    new_size : tuple[int, int]
        Width and height of the resized image.

    Returns
    -------
    tuple
        (top, bottom, left, right) padding sizes.
    """
    try:
        w, h = new_size
        right = int((EXPECTED_SIZE - w) / 2) + (EXPECTED_SIZE - w) % 2
        left  = int((EXPECTED_SIZE - w) / 2)
        top   = int((EXPECTED_SIZE - h) / 2) + (EXPECTED_SIZE - h) % 2
        bottom = int((EXPECTED_SIZE - h) / 2)
        return top, bottom, left, right
    except Exception as e:
        raise RuntimeError(f"Error computing margins: {e}") from e  

def run_segmentation(src: Union[np.ndarray, str], dst_dir: Optional[str] = None):
    """
    Run segmentation on either:
    - An in-memory image (np.ndarray, already BGR)
    - A directory tree (expects subfolders per label in LABELS)

    Returns
    -------
    np.ndarray or None
        - If `src` is an ndarray -> returns segmented image (H, W)
        - If `src` is a directory -> saves results to dst_dir and returns None
    """
    try:
        # --- Single-image mode ---
        if isinstance(src, np.ndarray):
            return image_segmentation(src)  # src is assumed BGR, returns (H,W)

        # --- Directory mode ---
        if isinstance(src, str) and os.path.isdir(src):
            if dst_dir is None:
                raise ValueError("dst_dir must be provided when processing a directory.")
            for label in LABELS:
                in_dir = os.path.join(src, label)
                out_dir = os.path.join(dst_dir, label)
                ensure_dir(out_dir)
                if not os.path.isdir(in_dir):
                    continue
                for name in os.listdir(in_dir):
                    path = os.path.join(in_dir, name)
                    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
                    if img is None:
                        continue
                    seg = image_segmentation(img)
                    cv2.imwrite(os.path.join(out_dir, name), seg)
            print(f"Segmentation complete: {src} -> {dst_dir}")
            return None

        raise TypeError("src must be either a NumPy image array (BGR) or a directory path.")
    except Exception as e:
        raise RuntimeError(f"Error during segmentation: {e}") from e

def _resize_and_pad(img_gray: np.ndarray) -> np.ndarray:
    """
    Resize and pad all images in a directory tree to EXPECTED_SIZE and save results.

    Parameters
    ----------
    img_gray : np.ndarray
        Grayscale image as a NumPy array.

    Returns
    -------
    resized and padded image as a NumPy array of shape (EXPECTED_SIZE, EXPECTED_SIZE).
    """
    try:
        h, w = img_gray.shape
        max_dim = max(h, w)
        sf = min(EXPECTED_SIZE / max_dim, 1.0)  # do not upscale
        new_w, new_h = int(w * sf), int(h * sf)
        resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if new_w < EXPECTED_SIZE or new_h < EXPECTED_SIZE:
            top, bottom, left, right = compute_margins((new_w, new_h))
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=0
            )
        return resized
    except Exception as e:
        raise RuntimeError(f"Error during resize and pad of images: {e}") from e


def resize_and_pad_images(src: Union[np.ndarray, str], dst_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Resize (keeping aspect ratio) and pad to EXPECTED_SIZE.
    
    Parameters
    ----------
    src : np.ndarray | str
        - If np.ndarray: single image (grayscale or BGR).
        - If str: directory containing LABELS subfolders.
    dst_dir : str, optional
        Output directory for directory mode.

    Returns
    -------
    np.ndarray or None
        - Returns processed image if `src` is np.ndarray
        - Returns None if processing a directory
    """
    try:
        # --- Single-image mode ---
        if isinstance(src, np.ndarray):
            if src.ndim == 3:  # BGR
                img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            elif src.ndim == 2:
                img_gray = src
            else:
                raise ValueError(f"Unsupported image shape: {src.shape}")
            return _resize_and_pad(img_gray)

        # --- Directory mode ---
        if isinstance(src, str) and os.path.isdir(src):
            if dst_dir is None:
                raise ValueError("dst_dir must be provided when processing a directory.")
            for label in LABELS:
                in_dir = os.path.join(src, label)
                out_dir = os.path.join(dst_dir, label)
                ensure_dir(out_dir)
                if not os.path.isdir(in_dir):
                    continue
                for fname in os.listdir(in_dir):
                    path = os.path.join(in_dir, fname)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    processed = _resize_and_pad(img)
                    cv2.imwrite(os.path.join(out_dir, fname), processed)
            print(f"Resize & pad complete: {src} -> {dst_dir}")
            return None

        raise TypeError("src must be either a NumPy image array or a directory path.")
    except Exception as e:
        raise RuntimeError(f"Error during resize and pad: {e}") from e