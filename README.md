# Driver Drowsiness Detection using a CNN

A real-time system that detects driver drowsiness by classifying eye state (open vs closed) from a webcam feed. Built as my MSc Big Data Analytics dissertation project at Sheffield Hallam University.

## Overview

Drowsy driving is a major cause of road accidents. This project trains a convolutional neural network (CNN) to recognise open and closed eyes, then uses it in a live webcam application that raises an on-screen alert when the driver's eyes stay closed for too long.

## How it works

**1. Training (`Drowsiness_detection.py`)**
- Loads open/closed eye images, converts them to grayscale, resizes to 80x80, applies histogram equalisation and scales to [0, 1]
- Stratified 70/15/15 train/validation/test split, done *before* augmentation so augmented copies never leak into validation or test data
- Data augmentation on the training set only (shifts, zoom, shear, rotation, horizontal flip)
- CNN: 4 Conv2D layers, BatchNormalization, MaxPooling, Dropout, 3 dense layers, sigmoid output giving P(eyes closed)
- Adam optimiser, binary cross-entropy, ModelCheckpoint + EarlyStopping on validation loss
- Decision threshold chosen by maximising F1 on the validation set
- Evaluation on the held-out test set: accuracy, precision, recall, F1, AUC, confusion matrix and ROC curve, saved to `results/`

**2. Live detection (`Camera_execution.py`)**
- Captures webcam frames with OpenCV
- Locates the eye using `face_recognition` facial landmarks and crops it
- Applies the same preprocessing as training (`common.py`)
- Shows "Eyes Open" / "Eyes Closed" and raises a **DRIVER SLEEPING** alert if the eyes stay closed for 1 second (configurable with `ALERT_AFTER_SECONDS`), so ordinary blinks don't trigger it

## Repository structure

```
.
├── Drowsiness_detection.py   # train + evaluate the model
├── Camera_execution.py       # real-time webcam demo
├── common.py                 # shared settings and preprocessing
├── requirements.txt
├── bestModel80.h5            # trained model (created by training)
├── threshold.json            # decision threshold (created by training)
├── results/                  # metrics and figures (created by training)
└── Dataset/train/            # NOT included - see below
    ├── Open_Eyes/
    └── Closed_Eyes/
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.9-3.11. `face_recognition` depends on `dlib`, which needs CMake and a C++ compiler to build on Windows (Visual Studio Build Tools). See the [face_recognition install notes](https://github.com/ageitgey/face_recognition#installation) if installation fails.

## Usage

1. Put your dataset in `Dataset/train/Open_Eyes` and `Dataset/train/Closed_Eyes` (_add dataset name and link here_)
2. Train: `python Drowsiness_detection.py`
3. Run the live demo: `python Camera_execution.py` (press Space, Q or Esc to exit)

## Results

Produced on the held-out test set and saved to `results/metrics.json`:

| Metric | Value |
|---|---|
| Accuracy | _add_ |
| Precision | _add_ |
| Recall | _add_ |
| F1-score | _add_ |
| AUC | _add_ |

Figures: `results/learning_curves.png`, `results/confusion_matrix.png`, `results/roc_curve.png`

## Limitations

- Only one eye (left, falling back to right) is classified per frame
- Performance depends on lighting, camera angle and glasses; the model was trained on a single dataset
- This is an academic prototype, not a certified safety system

## Future work

- Combine both eyes, plus yawning and head-pose cues
- Audible alarm and deployment on an embedded device
- Evaluate on more diverse datasets and real driving conditions

## Author

Nidheesh P. Pradeep Kumar
