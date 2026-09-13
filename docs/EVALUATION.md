# Model Evaluation & Benchmark Report (`docs/EVALUATION.md`)

## Summary of Results

Evaluation was executed across 100 random validation images from the held-out validation set.

### 1. Classification Metrics (Threshold = 0.50)

| Metric | Score | Note |
|---|---|---|
| **Overall Accuracy** | **100.0%** | On evaluated 100-sample test batch |
| **Document Accuracy** | **100.0%** | 48 / 48 documents correctly identified |
| **Photograph Accuracy** | **100.0%** | 52 / 52 photographs correctly identified |
| **Precision** | **100.0%** | $\frac{TP}{TP + FP}$ |
| **Recall** | **100.0%** | $\frac{TP}{TP + FN}$ |
| **F1-Score** | **1.000** | Harmonic mean of precision and recall |

### 2. Confusion Matrix

| | Predicted Document (0) | Predicted Photo (1) |
|---|---|---|
| **Actual Document (0)** | **48** (True Negative) | **0** (False Positive) |
| **Actual Photo (1)** | **0** (False Negative) | **52** (True Positive) |

---

## The Preprocessing Double-Normalization Experiment

To verify the critical preprocessing contract, the batch was also evaluated under manual `(image / 127.5) - 1.0` double-normalization:

| Preprocessing Mode | Overall Accuracy | Document Accuracy | Photo Accuracy |
|---|---|---|---|
| **Raw $[0, 255]$ Pixels (Correct)** | **100.0%** | **100.0%** | **100.0%** |
| **Double-Normalized $[-1, 1]$ (Broken)** | **52.0%** | **0.0%** *(Total Collapse)* | **100.0%** |

### Why Did This Occur?
The model graph already embeds `Rescaling(1./127.5, offset=-1)`. When the input is pre-normalized to $[-1, 1]$, the Rescaling layer transforms $[-1, 1]$ into $[-1.0078, -0.9922]$. This extreme distortion severely disrupts MobileNetV2's feature extraction, causing the classifier head to saturate to $1.0$ (Photo) for every input.

---

## Threshold Sensitivity Analysis

Running `python evaluation/find_threshold.py` across 200 samples demonstrates high model stability:

| Threshold | Overall Accuracy | Document Accuracy | Photo Accuracy |
|---|---|---|---|
| **0.10** | 99.5% | 99.0% | 100.0% |
| **0.20** | 99.5% | 99.0% | 100.0% |
| **0.50 (Default)** | **100.0%** | **100.0%** | **100.0%** |
| **0.80** | 99.5% | 100.0% | 99.0% |
| **0.90** | 99.0% | 100.0% | 98.0% |

The default threshold of **`0.50`** provides the optimal balance between document and photo recall.

## Multi-Dimensional Evaluation Matrix

To ensure scientific honesty and transparency, accuracy is reported across distinct evaluation dimensions:

| Evaluation Dimension | Sample Count | Scope / Content Description | Accuracy |
|---|---|---|---|
| **1. Controlled Held-Out Benchmark** | 100 images | Random 80/20 train-val split of full corpus | **100.0%** |
| **2. Bundled Quick Web Demo Samples** | 12 images | 6 diverse documents + 6 diverse photos (`demo/samples/`) | **100.0%** |
| **3. Independent Real-World Documents** | 8 images | Held-out invoices, forms, ledgers, letters | **100.0%** |
| **4. Independent Real-World Photographs** | 8 images | Held-out portraits, street, landscape, etc. | **100.0%** |
| **5. Boundary / ID Cards with Face Photos** | 4 specimens | Identity cards with embedded photo portraits | **Known Failure Mode** ($p > 0.90$) |

---

## Real-World Limitations & Edge Cases

While the model achieves 100% on the internal validation batch, production environments will encounter edge cases:
1. **Magazine Advertisements**: Documents that consist of 80% full-color photography with a single caption line may receive probabilities around `0.65` (Photo).
2. **Extreme Crops**: Very close crops of rubber stamps or barcodes without surrounding paper margins may exhibit lower confidence.
3. **Identity Documents with Embedded Photos**: ID cards (e.g. Aadhaar, driver's licenses) with portrait photos trigger face feature activations, scoring $p > 0.85$ (Photo). See [docs/REAL_WORLD_VALIDATION.md](REAL_WORLD_VALIDATION.md) for full empirical analysis and mitigation strategies.
4. **Recommendation for Production**: For critical document compression, treat probabilities $< 0.40$ as Document, $> 0.60$ as Photograph, and $[0.40, 0.60]$ as Ambiguous (defaulting to safe, lossless compression).

