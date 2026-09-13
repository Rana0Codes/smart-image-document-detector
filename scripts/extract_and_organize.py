"""
Extract and Organize Raw Archives into Smart Image & Document Detector Dataset.

Extracts:
1. RVL-CDIP documents into `dataset/document/`
2. General Purpose photos into `dataset/photo/`
Converts non-JPEG/PNG formats (e.g. TIFF) to clean RGB JPEGs.
"""

import argparse
import os
from pathlib import Path
import shutil
import zipfile
from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_EXTENSIONS = VALID_EXTENSIONS.union({".tif", ".tiff", ".bmp", ".gif", ".jpe"})


def is_image(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


def extract_archive(zip_path: Path, output_dir: Path, prefix: str):
    if not zip_path.exists():
        print(f"Archive not found: {zip_path} (Skipping)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} to {output_dir.name}...")

    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        for i, file_info in enumerate(z.infolist()):
            if file_info.is_dir() or not is_image(file_info.filename):
                continue

            ext = os.path.splitext(file_info.filename)[1].lower()
            try:
                with z.open(file_info) as f:
                    if ext in VALID_EXTENSIONS:
                        dest_path = output_dir / f"{prefix}_{i}{ext}"
                        with open(dest_path, "wb") as dest_file:
                            dest_file.write(f.read())
                    else:
                        img = Image.open(f)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        dest_path = output_dir / f"{prefix}_{i}.jpg"
                        img.save(dest_path, "JPEG")
                    extracted += 1
            except Exception as e:
                print(f"Skipping {file_info.filename}: {e}")

    print(f"Extracted {extracted} images to {output_dir}.")


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Extract and organize raw zip datasets.")
    parser.add_argument(
        "--docs-zip",
        type=str,
        default="the rvl cdip dataset.zip",
        help="Path to the RVL-CDIP documents zip file.",
    )
    parser.add_argument(
        "--photos-zip",
        type=str,
        default="General Purpose Image Dataset.zip",
        help="Path to the General Purpose photos zip file.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(project_root / "dataset"),
        help="Output dataset base directory.",
    )
    args = parser.parse_args()

    base_dir = Path(args.dataset_dir)
    docs_dir = base_dir / "document"
    photos_dir = base_dir / "photo"

    extract_archive(Path(args.docs_zip), docs_dir, "doc")
    extract_archive(Path(args.photos_zip), photos_dir, "photo")
    print("\nExtraction and organization complete!")


if __name__ == "__main__":
    main()
