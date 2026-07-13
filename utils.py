"""
Utility module for the CheXpert chest X-ray image classification model.

This module provides logging configurations, image preprocessing utilities,
the TensorFlow/Keras model reconstruction, and inference wrappers.
"""

import time
import logging
from typing import Dict, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CheXpertUtils")

# CheXpert target pathology labels in training column order
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]


def build_model(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Sequential:
    """
    Reconstructs the model architecture matching the trained model.

    Args:
        input_shape (Tuple[int, int, int]): Expected shape of input images.

    Returns:
        Sequential: Keras sequential model containing EfficientNetB0.
    """
    logger.info("Instantiating EfficientNetB0 base model (include_top=False, weights=None)...")
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,  # Set to None to prevent downloading pre-trained weights
        input_shape=input_shape,
        pooling="max"
    )
    # The base model is frozen in feature extraction phase but full architecture is saved.
    base_model.trainable = False

    logger.info("Constructing classification head and Sequential wrapper...")
    model = Sequential([
        base_model,
        BatchNormalization(),
        Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.2),
        Dense(len(CHEXPERT_LABELS), activation="sigmoid")
    ])

    return model


def load_model_weights(model: Sequential, weights_path: str) -> Sequential:
    """
    Loads trained weights into the reconstructed model.

    Args:
        model (Sequential): Reconstructed Keras model.
        weights_path (str): Path to the H5 weight file.

    Returns:
        Sequential: Loaded model.
    """
    logger.info(f"Loading weights from {weights_path}...")
    start_time = time.time()
    try:
        model.load_weights(weights_path)
        latency = time.time() - start_time
        logger.info(f"Weights loaded successfully in {latency:.3f} seconds.")
    except Exception as e:
        logger.error(f"Failed to load model weights from {weights_path}: {e}")
        raise e
    return model


def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Loads a PIL Image, resizes, converts to RGB if needed, and applies Keras preprocessing.

    Args:
        image (Image.Image): Uploaded raw chest X-ray image.
        target_size (Tuple[int, int]): Required input size for EfficientNetB0.

    Returns:
        np.ndarray: Preprocessed image tensor ready for model input.
    """
    logger.info(f"Starting image preprocessing. Input mode: {image.mode}, original size: {image.size}")
    start_time = time.time()

    # Convert to RGB (3 channels) if grayscale or RGBA
    if image.mode != "RGB":
        logger.info(f"Converting image from mode '{image.mode}' to 'RGB'")
        image = image.convert("RGB")

    # Resize image using Bilinear resampling (highly robust fallback for older pillow versions)
    try:
        resample_filter = Image.Resampling.BILINEAR
    except AttributeError:
        resample_filter = Image.BILINEAR

    image = image.resize(target_size, resample_filter)
    logger.info(f"Resized image to {target_size}")

    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)

    # Add batch dimension: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Apply EfficientNet preprocessing (expects values [0, 255])
    img_array = preprocess_input(img_array)

    latency = time.time() - start_time
    logger.info(f"Preprocessing completed in {latency:.4f} seconds.")
    return img_array


def run_inference(model: Sequential, img_array: np.ndarray) -> Dict[str, float]:
    """
    Runs model prediction and outputs a dictionary mapping labels to probabilities.

    Args:
        model (Sequential): Loaded Keras model.
        img_array (np.ndarray): Preprocessed image array.

    Returns:
        Dict[str, float]: Label to predicted probability mapping.
    """
    logger.info("Running model inference...")
    start_time = time.time()
    try:
        predictions = model.predict(img_array, verbose=0)
        latency = time.time() - start_time
        logger.info(f"Inference run completed in {latency:.4f} seconds.")

        # Flatten outputs and convert to native float type
        probs = predictions[0].tolist()

        # Map predictions to label names
        results = {label: float(prob) for label, prob in zip(CHEXPERT_LABELS, probs)}
        return results
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise e
