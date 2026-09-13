# Model Architecture (`docs/MODEL.md`)

## High-Level Overview

**Smart Image & Document Detector** is a lightweight binary image classifier designed for client-side browser execution. It identifies whether an image is a **Document (Class 0)** or a **Photograph (Class 1)** to dynamically toggle downstream processing pipelines (e.g., OCR and lossless text preservation vs lossy WebP/AVIF compression for bandwidth savings).

```
             ┌────────────────────────────────────┐
             │       Input Image (RGB)            │
             │         224 × 224 × 3              │
             │         Range: [0, 255]            │
             └──────────────────┬─────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────┐
             │         Embedded Rescaling         │
             │    x = (input / 127.5) - 1.0       │
             │         Range: [-1, 1]             │
             └──────────────────┬─────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────┐
             │        MobileNetV2 Backbone        │
             │   ImageNet Pre-trained (Frozen)    │
             │      Output Shape: 7 × 7 × 1280    │
             └──────────────────┬─────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────┐
             │       GlobalAveragePooling2D       │
             │         Output Shape: 1280         │
             └──────────────────┬─────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────┐
             │            Dropout(0.2)            │
             └──────────────────┬─────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────┐
             │        Dense(1, Sigmoid)           │
             │     Probability Scalar p ∈ [0, 1]  │
             └────────────────────────────────────┘
```

---

## Architectural Breakdown

### 1. Embedded Input Rescaling Layer
- **Layer**: `tf.keras.layers.Rescaling(1./127.5, offset=-1)`
- **Purpose**: MobileNetV2 was trained on ImageNet with inputs scaled to $[-1.0, 1.0]$. Embedding this operation directly into the Keras model computational graph ensures that all external callers (JavaScript, Python, C++, WebGL) simply pass standard raw pixel values in $[0, 255]$.
- **Contract Rule**: External inference code must **never** divide by 127.5 or subtract 1.0 before feeding the model.

### 2. Feature Extractor: MobileNetV2
- **Base Architecture**: MobileNetV2 (depthwise separable convolutions with inverted residuals and linear bottlenecks).
- **Pretraining**: ImageNet-1k classification weights.
- **Trainability**: `base_model.trainable = False`. All feature extractor convolutional filters are frozen, acting as a robust, non-degrading feature extractor while minimizing training time and overfitting.

### 3. Classification Head
- **`GlobalAveragePooling2D`**: Collapses the $7 \times 7 \times 1280$ feature maps into a 1D vector of 1280 features by spatial average pooling across height and width.
- **`Dropout(0.2)`**: Regularization layer dropping 20% of activations during training to prevent co-adaptation.
- **`Dense(1, activation='sigmoid')`**: Single output neuron with logistic sigmoid activation computing the posterior probability $P(\text{Photo} \mid \text{Image})$.

---

## Output Semantics & Interpretation

The model output $p$ is a single floating-point number in $[0.0, 1.0]$:

$$\hat{y} = \begin{cases} \text{Document (Class 0)} & \text{if } p < 0.50 \\ \text{Photograph (Class 1)} & \text{if } p \ge 0.50 \end{cases}$$

### Confidence Metric
$$\text{Confidence}(p) = \begin{cases} (1 - p) \times 100\% & \text{if } p < 0.50 \\ p \times 100\% & \text{if } p \ge 0.50 \end{cases}$$

### Example Predictions
- $p = 0.0021 \rightarrow$ **Document** with **99.8% confidence**.
- $p = 0.0450 \rightarrow$ **Document** with **95.5% confidence**.
- $p = 0.9850 \rightarrow$ **Photograph** with **98.5% confidence**.
- $p = 0.5200 \rightarrow$ **Ambiguous / Borderline Photograph** with **52.0% confidence**.

---

## Domain Adaptation & Real-World Caveats

While the model achieves $>99.9\%$ accuracy on standard scanned document archives and outdoor/indoor photography, developers should note an important domain edge case:
- **ID Cards / Passports**: Documents featuring embedded human portrait photos on colored plastic substrates often activate face and texture feature detectors in the MobileNetV2 backbone, producing photograph scores ($p > 0.90$).
- Refer to [docs/REAL_WORLD_VALIDATION.md](REAL_WORLD_VALIDATION.md) for full empirical analysis and recommended architectural mitigation techniques (e.g., text density heuristics or multi-task finetuning).
