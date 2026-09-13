# Training & Retraining Guide (`docs/TRAINING.md`)

This guide explains how the **Smart Image & Document Detector** model was trained and how you can reproduce or fine-tune it with additional data.

## Training Configuration & Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| **Base Architecture** | `MobileNetV2` | Pretrained on ImageNet |
| **Base Weights** | Frozen (`trainable=False`) | Feature extractor kept fixed |
| **Input Dimensions** | $224 \times 224 \times 3$ | RGB channels |
| **Batch Size** | `32` | Mini-batch gradient descent |
| **Epochs** | `5` | Convergence reached in 5 epochs |
| **Optimizer** | `Adam` | Adaptive moment estimation |
| **Learning Rate** | `0.001` | Default Adam learning rate |
| **Loss Function** | `binary_crossentropy` | Log loss for binary classification |
| **Validation Split** | `0.20` (20%) | 80% train / 20% validation |
| **Seed** | `123` | Deterministic dataset split |
| **Regularization** | `Dropout(0.2)` | Prevents head overfitting |
| **Checkpoints** | `ModelCheckpoint` | Saves weights after every epoch |

---

## Step-by-Step Retraining Procedure

### 1. Environment Setup
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Organize your images into `dataset/`:
```
dataset/
├── document/
│   ├── image001.jpg
│   └── ...
└── photo/
    ├── image001.jpg
    └── ...
```

Run dataset validation to clean any corrupt files:
```bash
python scripts/validate_dataset.py
```

### 3. Launch Training
Run the training script:
```bash
python training/train_model.py --epochs 5 --batch-size 32
```

### 4. Export the Model
Re-export the SavedModel and Float32 TensorFlow.js formats:
```bash
python training/export_model.py
```

Then generate the UINT8 quantized model for browser deployment:
```bash
python training/export_quantized.py
```

---

## Continuing Training from Checkpoints

The training script automatically detects if a checkpoint exists at `training/checkpoints/cp.weights.h5`:
- If found, it loads the saved weights and resumes training from the checkpoint.
- If not found, it initializes weights randomly for the head and uses ImageNet weights for MobileNetV2.

To start fresh from scratch, delete or rename `training/checkpoints/cp.weights.h5`.
