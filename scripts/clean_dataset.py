"""
Clean Dataset Utility for Smart Image & Document Detector.

Scans the dataset directory, verifies that each image is structurally intact,
and converts non-standard images to clean RGB JPEG files. Any unreadable or
irreparably corrupt files are safely removed.
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image


def clean_dataset(data_dir: Path):
    print(f"Scanning and cleaning dataset at: {data_dir.resolve()}")
    if not data_dir.exists():
        print(f"Directory does not exist: {data_dir}")
        return

    removed = 0
    converted = 0
    verified = 0

    for root, _, files in os.walk(data_dir):
        for f in files:
            file_path = Path(root) / f
            try:
                # 1. Structural verification
                with Image.open(file_path) as img:
                    img.verify()

                # 2. Re-open to standardize color space and format
                with Image.open(file_path) as img:
                    if img.mode != "RGB" or img.format != "JPEG":
                        rgb_img = img.convert("RGB")
                        # Save in place as standard JPEG
                        rgb_img.save(file_path, "JPEG", quality=95)
                        converted += 1
                    else:
                        verified += 1

            except Exception as e:
                print(f"Removing corrupt image: {file_path.name} ({e})")
                try:
                    os.remove(file_path)
                    removed += 1
                except OSError as rm_err:
                    print(f"Failed to delete {file_path}: {rm_err}")

    print("-" * 50)
    print(f"Clean complete: {verified} verified, {converted} converted to RGB, {removed} removed.")


def main():
    parser = argparse.ArgumentParser(description="Sanitize dataset images.")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(project_root / "dataset"),
        help="Target dataset directory.",
    )
    args = parser.parse_args()
    clean_dataset(Path(args.data_dir))


if __name__ == "__main__":
    main()
