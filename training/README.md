# Training Pipeline (`training/`)

This directory contains the end-to-end training and export scripts for the **Smart Image & Document Detector** binary classifier.

## Overview

- **Backbone**: MobileNetV2 (Pre-trained on ImageNet, base layers frozen)
- **Input Dimensions**: $224 \times 224 \times 3$ RGB
- **Input Value Range**: Raw $[0, 255]$ pixel values (internal `Rescaling(1./127.5, offset=-1)` handles normalization)
- **Classifier Head**:
  ```
  GlobalAveragePooling2D
  Dropout(0.2)
  Dense(1, activation='sigmoid')
  ```
- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `Adam(learning_rate=0.001)`
- **Classes**:
  - `0`: Document
  - `1`: Photograph

---

## Retraining the Model

### Step 1: Prepare the Dataset
Ensure images are structured into `dataset/document/` and `dataset/photo/`.
Run verification:
```bash
python scripts/validate_dataset.py
```

### Step 2: Run Training
Execute `train_model.py`:
```bash
python training/train_model.py --epochs 5 --batch-size 32
```

This will:
1. Split the dataset 80% for training and 20% for validation (seed 123).
2. Save training checkpoints to `training/checkpoints/cp.weights.h5`.
3. Export the final model to `models/tensorflow_savedmodel/`.

---

## Exporting for Web Deployment

### 1. Export Float32 TensorFlow.js Model (~8.8 MB)
```bash
python training/export_model.py
```

### 2. Export UINT8 Quantized TensorFlow.js Model (~2.2 MB - Recommended)
```bash
python training/export_quantized.py
```
This produces an 8-bit quantized graph model in `models/tfjs_uint8/` ready for browser inference.
