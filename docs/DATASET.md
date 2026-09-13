# Dataset Documentation (`docs/DATASET.md`)

## Overview

The **Smart Image & Document Detector** dataset contains **11,395 images** categorized into two classes:
- **Class 0: `document`** (5,000 images, ~499 MB)
- **Class 1: `photo`** (6,395 images, ~1,355 MB)

---

## Dataset Provenance & Sources

### 1. Document Class (RVL-CDIP)
- **Primary Source**: Ryerson Vision Lab Complex Document Information Processing (RVL-CDIP) benchmark.
- **Underlying Archive**: The Truth Tobacco Industry Documents archive collected by the University of California, San Francisco (UCSF) Library.
- **Nature of Documents**: Scanned physical paperwork from 1960–2000, including invoices, tax returns, accounting ledgers, purchase receipts, typed memos, and legal notices.
- **Licensing & Legal Status**: Documents produced during legal discovery and government proceedings in the United States are in the public domain. Research distribution is provided for open computer vision evaluation.

### 2. Photograph Class (General Purpose Image Dataset)
- **Primary Source**: Curated photography from open-source computer vision datasets and open media archives (Wikimedia Commons, Unsplash open license).
- **Nature of Photos**: Digital camera photos and scans of portraits, wildlife, pets, outdoor landscapes, vehicles, and architecture.

---

## Data Sanitization & Cleaning Pipeline

During raw extraction from zip archives, raw images passed through a 4-step pipeline:
1. **Format Unification**: Documents stored as TIFF (`.tif`) were converted to standard 24-bit RGB JPEGs (`.jpg`).
2. **Color Mode Standardization**: Greyscale (mode `L`), palletized (mode `P`), and transparency (mode `RGBA`) images were converted to 3-channel standard `RGB`.
3. **Integrity Validation**: Each file was opened with `Image.open().verify()` to discard corrupted bitstreams.
4. **Duplicate Removal**: Identical images were filtered using MD5 checksum comparisons.

---

## Directory Structure

```
dataset/
├── README.md               # Quick dataset guide
├── samples/                # Bundled in Git for instant CI/testing (50 images)
│   ├── document/           # 25 sample document images
│   └── photo/              # 25 sample photo images
├── document/               # Full corpus (5,000 images, ignored in git)
└── photo/                  # Full corpus (6,395 images, ignored in git)
```

### Sample Sets in Repository
To avoid any ambiguity between sample directories:
- **`dataset/samples/`** (50 images): Bundled smoke-test dataset (25 documents, 25 photos) used by `scripts/validate_dataset.py --samples-only` and offline unit tests.
- **`demo/samples/`** (12 images): Curated showcase images (6 documents, 6 photos) used by the web browser demo GUI (`demo/index.html`).
- **`evaluation/real_world_validation/`** (16 images + 4 boundary cases): Independent held-out validation suite used by `evaluation/verify_real_world.py`.

---

## Obtaining the Full Dataset

Because Git is optimized for source code rather than multi-gigabyte binary datasets, the full 1.85 GB image corpus is excluded from git commits.

To recreate the full dataset:
1. Download the RVL-CDIP subset and General Purpose Photo archives.
2. Place the archives in the project root:
   - `the rvl cdip dataset.zip`
   - `General Purpose Image Dataset.zip`
3. Run the automated extractor:
   ```bash
   python scripts/extract_and_organize.py
   ```
4. Run validation:
   ```bash
   python scripts/validate_dataset.py
   ```
