# Dataset Guide (`dataset/`)

This directory contains information, sample images, and specifications for the **Smart Image & Document Detector** training and evaluation datasets.

## Class Specification

| Class Label | Index | Target Types | Compression Target |
|---|---|---|---|
| **`document`** | `0` | Receipts, invoices, tax forms, passports, letters, handwritten notes, IDs, scanned text documents | **Lossless / Text-preserving** |
| **`photo`** | `1` | Portraits, landscapes, nature, animals, outdoor scenes, camera photographs | **High-efficiency Lossy** |

---

## Dataset Statistics

- **Total Images**: 11,395 images
- **Document Images**: 5,000 images (~499 MB)
- **Photograph Images**: 6,395 images (~1,355 MB)
- **Total Storage Size**: ~1.85 GB
- **Image Format**: RGB JPEG and PNG
- **Target Resolution**: Resized at training/inference time to $224 \times 224$ pixels.

---

## Repository Storage & Distribution Strategy

### 1. Bundled Smoke-Test Sample Set (`dataset/samples/` - 50 Images)
To ensure anyone cloning this repository can immediately run evaluation, tests, and smoke validation without downloading gigabytes of data, the repository includes a curated smoke-test set:
- `dataset/samples/document/` (25 sample document images)
- `dataset/samples/photo/` (25 sample photograph images)

To validate the smoke-test set:
```bash
python scripts/validate_dataset.py --samples-only
```

*(Note: The web demo in `demo/samples/` contains 12 separate showcase images, validatable via `python scripts/validate_dataset.py --demo-samples`.)*

### 2. Full Training Dataset (1.85 GB)
To comply with GitHub repository size guidelines (< 1 GB recommended) and maintain rapid repository cloning speeds, the complete 1.85 GB image corpus is excluded from git commits via `.gitignore`:
```gitignore
dataset/document/
dataset/photo/
```

If you wish to train or reproduce on the full 11,395 image set on your local machine:
1. Obtain the **RVL-CDIP** dataset and **General Purpose Image Dataset** archives.
2. Run the extraction utility:
   ```bash
   python scripts/extract_and_organize.py --docs-zip "the rvl cdip dataset.zip" --photos-zip "General Purpose Image Dataset.zip"
   ```
3. Sanitize and validate:
   ```bash
   python scripts/clean_dataset.py
   python scripts/validate_dataset.py
   ```

---

## Data Provenance & Licensing

### Document Source: RVL-CDIP Dataset
- **Origin**: Ryerson Vision Lab Complex Document Information Processing (RVL-CDIP) benchmark, sampled from the Truth Tobacco Industry Documents archive collected by the University of California, San Francisco (UCSF).
- **Content**: Real-world scanned receipts, invoices, forms, memos, reports, and letters.
- **Terms**: Open research and educational use. Litigation discovery documents are in the public domain.

### Photograph Source: General Purpose Image Dataset
- **Origin**: Curated collections of general photography from open-source computer vision datasets and Wikimedia Commons.
- **Content**: Nature, urban settings, domestic animals, and portraits.
- **Terms**: Creative Commons / Open Access.

---

## Adding Your Own Images

To augment the dataset with your own custom images:
1. Place document images into `dataset/document/`.
2. Place photograph images into `dataset/photo/`.
3. Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`.
4. Ensure images are RGB (3-channel).
5. Run the validator:
   ```bash
   python scripts/validate_dataset.py
   ```
6. Retrain the model:
   ```bash
   python training/train_model.py
   ```
