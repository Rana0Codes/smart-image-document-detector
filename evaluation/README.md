# Model Evaluation (`evaluation/`)

This directory provides scripts to evaluate, test, and benchmark the **Smart Image & Document Detector** model.

## Evaluation Scripts

| Script | Purpose |
|---|---|
| [`verify_real_world.py`](verify_real_world.py) | Validates diverse real-world samples (invoices, forms, reports, diverse photos) and provides domain shift diagnostic analysis. |
| [`verify_batch.py`](verify_batch.py) | Evaluates batch accuracy on validation images, reporting Accuracy, Document Accuracy, Photo Accuracy, Precision, Recall, F1 Score, and Confusion Matrix. |
| [`find_threshold.py`](find_threshold.py) | Sweeps classification thresholds between `0.05` and `0.95` to locate the optimal decision boundary. |
| [`test_preprocessing.py`](test_preprocessing.py) | Inspects a single test image and contrasts predictions under raw `[0, 255]` vs double-normalized `[-1, 1]` inputs. |
| [`evaluate_quantization.py`](evaluate_quantization.py) | Analyzes model sizes, shard counts, and network bandwidth savings for Float32 vs UINT8 models. |

---

## Verified Evaluation Benchmarks

Running `python evaluation/verify_batch.py --count 100`:

```
=================================================================
EVALUATION RESULTS (100 Validation Samples)
=================================================================
[A] Correct Preprocessing: Raw [0, 255] Pixels (Embedded Rescaling)
  Overall Accuracy:      100.0%
  Document Accuracy:     100.0%
  Photograph Accuracy:   100.0%
  Precision:             100.0%
  Recall:                100.0%
  F1 Score:              100.0%

  Confusion Matrix:
    True Negative (Doc -> Doc):    48
    False Positive (Doc -> Photo): 0
    False Negative (Photo -> Doc): 0
    True Positive (Photo -> Photo):52

[B] Broken Preprocessing: Double-Normalized [-1, 1] Pixels
  Overall Accuracy:      52.0%
  Document Accuracy:     0.0%  <-- COLLAPSED!
  Photograph Accuracy:   100.0%
```

> [!NOTE]
> **Important Note on Real-World Accuracy**:
> While the model achieves 100% accuracy on this validation batch, real-world images with unusual lighting, complex graphic design, blended photo-text layouts, or extreme crops may exhibit ambiguous scores. In production, scores within $[0.40, 0.60]$ should be treated as uncertain or fall back to conservative compression.
