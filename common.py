"""Shared settings and preprocessing for training and live detection.

Both scripts import from here so the model sees identically processed
images during training and in the webcam demo.
"""
import json
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

IMG_SIZE = 80
MODEL_PATH = BASE_DIR / "bestModel80.h5"
THRESHOLD_PATH = BASE_DIR / "threshold.json"
DEFAULT_THRESHOLD = 0.5

# Label convention used everywhere: 1 = eyes closed, 0 = eyes open.
# The model outputs the probability that the eye is CLOSED.
CLOSED, OPEN = 1, 0


def preprocess_eye(gray):
    """Convert a uint8 grayscale eye image of any size into model input.

    Steps: resize to 80x80 -> histogram equalisation (standardises
    lighting) -> scale to [0, 1] -> add a channel axis.
    Returns a float32 array of shape (80, 80, 1).
    """
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = cv2.equalizeHist(gray)
    return (gray.astype("float32") / 255.0)[..., np.newaxis]


def load_threshold():
    """Decision threshold chosen on the validation set during training.

    Falls back to 0.5 if training has not been run yet.
    """
    try:
        with open(THRESHOLD_PATH) as f:
            return float(json.load(f)["threshold"])
    except (FileNotFoundError, KeyError, ValueError):
        return DEFAULT_THRESHOLD
