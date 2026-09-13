"""
Dataset Validation Script for Smart Image & Document Detector.

Scans dataset directories to verify:
1. File readability and integrity (detects corrupt images)
2. Color modes (ensures 3-channel RGB)
3. Image dimensions
4. Duplicate detection via MD5 hashing
5. Class distribution and directory structure
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from PIL import Image


def calculate_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file for duplicate detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_directory(dir_path: Path, class_name: str, check_duplicates: bool = True):
    """Validate all images inside a specific class directory."""
    print(f"\n--- Validating Class: '{class_name}' ({dir_path}) ---")
    if not dir_path.exists():
        print(f"ERROR: Directory does not exist: {dir_path}")
        return False, {}

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
def validate_files(file_list: list, class_name: str, check_duplicates: bool = True):
    """Validate a list of image file paths for integrity, RGB mode, and duplicates."""
    print(f"\n--- Validating Class: '{class_name}' ({len(file_list)} files) ---")
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    total_files = len(file_list)

    if total_files == 0:
        print("WARNING: No files provided for validation.")
        return False, {}

    valid_count = 0
    corrupt_count = 0
    converted_mode_count = 0
    seen_hashes = {}
    duplicates = []
    formats = {}
    modes = {}

    for file_path in file_list:
        ext = file_path.suffix.lower()
        if ext not in valid_extensions:
            print(f"Unsupported extension: {file_path.name}")
            corrupt_count += 1
            continue

        try:
            with Image.open(file_path) as img:
                img.verify()

            with Image.open(file_path) as img:
                fmt = img.format or "UNKNOWN"
                mode = img.mode
                formats[fmt] = formats.get(fmt, 0) + 1
                modes[mode] = modes.get(mode, 0) + 1

                if mode != "RGB":
                    converted_mode_count += 1

            if check_duplicates:
                file_hash = calculate_md5(file_path)
                if file_hash in seen_hashes:
                    duplicates.append((file_path.name, seen_hashes[file_hash]))
                else:
                    seen_hashes[file_hash] = file_path.name

            valid_count += 1

        except Exception as e:
            print(f"Corrupt image detected: {file_path.name} - Error: {e}")
            corrupt_count += 1

    stats = {
        "class": class_name,
        "total": total_files,
        "valid": valid_count,
        "corrupt": corrupt_count,
        "non_rgb": converted_mode_count,
        "duplicates": len(duplicates),
        "formats": formats,
        "modes": modes,
    }

    print(f"Total files: {total_files}")
    print(f"Valid readable images: {valid_count}")
    print(f"Corrupt / Unreadable: {corrupt_count}")
    print(f"Non-RGB mode images: {converted_mode_count}")
    print(f"Image formats: {formats}")
    print(f"Color modes: {modes}")
    if duplicates:
        print(f"Duplicate images found: {len(duplicates)}")
    else:
        print("No duplicate images found.")

    is_healthy = corrupt_count == 0 and valid_count > 0
    return is_healthy, stats


def validate_directory(dir_path: Path, class_name: str, check_duplicates: bool = True):
    """Validate all images inside a specific class directory."""
    print(f"\n--- Validating Class: '{class_name}' ({dir_path}) ---")
    if not dir_path.exists():
        print(f"ERROR: Directory does not exist: {dir_path}")
        return False, {}

    all_files = [p for p in dir_path.iterdir() if p.is_file()]
    return validate_files(all_files, class_name, check_duplicates)


def main():
    parser = argparse.ArgumentParser(description="Validate dataset integrity for Smart Image & Document Detector.")
    project_root = Path(__file__).resolve().parent.parent
    default_dataset = project_root / "dataset"
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(default_dataset),
        help="Path to dataset directory containing 'document' and 'photo' subdirectories.",
    )
    parser.add_argument(
        "--samples-only",
        action="store_true",
        help="Validate bundled smoke test samples in 'dataset/samples' (50 images).",
    )
    parser.add_argument(
        "--demo-samples",
        action="store_true",
        help="Validate curated web demo samples in 'demo/samples' (12 images).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Smart Image & Document Detector Dataset Validator")
    print("=" * 60)

    if args.demo_samples:
        demo_dir = project_root / "demo" / "samples"
        print(f"Target Mode: Web Demo Samples ({demo_dir.resolve()})")
        print("=" * 60)
        all_samples = sorted([p for p in demo_dir.iterdir() if p.is_file()])
        doc_files = [p for p in all_samples if p.name.startswith("document_")]
        photo_files = [p for p in all_samples if p.name.startswith("photo_")]

        doc_ok, doc_stats = validate_files(doc_files, "document (class 0)")
        photo_ok, photo_stats = validate_files(photo_files, "photo (class 1)")
    else:
        base_dir = Path(args.dataset_dir)
        if args.samples_only:
            base_dir = base_dir / "samples"

        print(f"Target Directory: {base_dir.resolve()}")
        print("=" * 60)

        doc_dir = base_dir / "document"
        photo_dir = base_dir / "photo"

        doc_ok, doc_stats = validate_directory(doc_dir, "document (class 0)")
        photo_ok, photo_stats = validate_directory(photo_dir, "photo (class 1)")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    total_images = doc_stats.get("valid", 0) + photo_stats.get("valid", 0)
    print(f"Total Valid Images: {total_images}")
    print(f"  - Document: {doc_stats.get('valid', 0)}")
    print(f"  - Photo:    {photo_stats.get('valid', 0)}")

    if doc_ok and photo_ok:
        print("\nSUCCESS: Dataset passed all integrity checks!")
        sys.exit(0)
    else:
        print("\nFAILURE: One or more issues found in dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
