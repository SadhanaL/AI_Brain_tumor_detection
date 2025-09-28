from typing import Dict
import os
import numpy as np
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from CNNetwork import CNNetwork  
from utils import run_segmentation, resize_and_pad_images
from config import MODEL_PATH

# Global model + expected input shape
model: CNNetwork | None = None
INPUT_SHAPE: tuple[int, int] | None = None  # (H, W)


class PredictResponse(BaseModel):
    predicted_label: str
    probabilities: Dict[str, float]

# lifespan (startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.

    Loads the trained CNNetwork model at startup and sets the expected input shape.
    Releases resources and resets global variables on shutdown.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the model file is missing or loading fails.
    """
    global model, INPUT_SHAPE

    # STARTUP
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file '{MODEL_PATH}' not found. "
            "Train and save the model first using main.py."
        )
    try:
        model = CNNetwork.load(MODEL_PATH)
        INPUT_SHAPE = (model.input_h, model.input_w)
        print(f"Model loaded. Expecting grayscale {INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}")
    except Exception as e:
        # Fail fast on startup if model can't load
        raise RuntimeError(f"Failed to load model from '{MODEL_PATH}': {e}") from e

    # Hand control to the app
    try:
        yield
    finally:
        # SHUTDOWN - clean up resources 
        model = None
        INPUT_SHAPE = None
        print("Shutting down... resources released.")

app = FastAPI(
    title="CNNetwork Classifier",
    version="1.0.0",
    description="Simple API for a fixed-kernel CNN (conv + ReLU + maxpool + softmax). Upload an image to get a yes/no classification.",
    lifespan=lifespan
)

# routes
@app.get("/", include_in_schema=False)
def root():
    """
    Redirect the root URL to the Swagger UI documentation.

    Parameters
    ----------
    None

    Returns
    -------
    RedirectResponse
        Redirects to /docs for interactive API documentation.
    """
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=PredictResponse, summary="Classify an uploaded image")
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded image using the loaded CNNetwork model. The image is first segmented, resized, and normalized before prediction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Basic MIME guard (for browsers, curl, Postman, etc.)
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        content = await file.read()
        buf = np.frombuffer(content, np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR) 
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image bytes.")
        
        segmented = run_segmentation(img_bgr)
        resized = resize_and_pad_images(segmented)
        normalized = resized.astype(np.float32) / 255.0
        probs = model.forward_prop(normalized)  # (2,) array of float
        p_no, p_yes = float(probs[0]), float(probs[1])
        label = "yes" if p_yes >= p_no else "no"

        return PredictResponse(
            predicted_label=label,
            probabilities={"no": p_no, "yes": p_yes},
        )
    except HTTPException:
        raise
    except Exception as e:
        # Surface a clean error to client
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")
