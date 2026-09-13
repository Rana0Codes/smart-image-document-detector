# Utility Scripts (`scripts/`)

This directory contains standalone utility scripts for dataset preparation, verification, and testing support.

| Script | Purpose |
|---|---|
| [`validate_dataset.py`](validate_dataset.py) | Verifies image structural integrity, formats (JPEG/PNG/WebP), color modes (RGB), detects corrupted files, and checks duplicates. |
| [`clean_dataset.py`](clean_dataset.py) | Sanitizes an image collection by verifying readability and converting non-RGB or unusual modes into clean RGB JPEGs. |
| [`extract_and_organize.py`](extract_and_organize.py) | Extracts raw zip archives (RVL-CDIP documents and General Purpose photos), converts formats (e.g. TIFF to JPEG), and organizes them into `dataset/document/` and `dataset/photo/`. |
| [`download_samples.py`](download_samples.py) | Downloads a small set of sample images from open Wikimedia Commons repositories for immediate out-of-the-box evaluation. |
| [`generate_dummy_data.py`](generate_dummy_data.py) | Generates synthetic sample images for CI/CD test runs without requiring external downloads. |
| [`build_real_world_validation_set.py`](build_real_world_validation_set.py) | Populates the independent real-world validation suite and boundary ID card specimens. |

### Quick Examples

Validate the bundled sample images:
```bash
python scripts/validate_dataset.py --samples-only
```

Validate the full dataset:
```bash
python scripts/validate_dataset.py
```

Clean a downloaded directory of images:
```bash
python scripts/clean_dataset.py --data-dir dataset
```
