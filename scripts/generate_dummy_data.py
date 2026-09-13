"""
Generate Synthetic Dummy Dataset.

Generates random synthetic noise images for document and photo classes to allow
smoke-testing the training and evaluation pipelines in environments where the
real dataset is not downloaded.
"""

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image


def generate_dummy_data(output_dir: Path, samples_per_class: int = 30):
    docs_dir = output_dir / "document"
    photos_dir = output_dir / "photo"

    docs_dir.mkdir(parents=True, exist_ok=True)
    photos_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {samples_per_class} synthetic images per class at {output_dir}...")

    for i in range(samples_per_class):
        # Documents: mostly high-value pixels (light backgrounds like paper)
        doc_arr = np.random.randint(200, 255, (224, 224, 3), dtype=np.uint8)
        # Add a few dark horizontal streaks to simulate text lines
        for line_y in range(40, 200, 20):
            doc_arr[line_y : line_y + 4, 30:190] = np.random.randint(0, 50, (4, 160, 3), dtype=np.uint8)
        doc_img = Image.fromarray(doc_arr)
        doc_img.save(docs_dir / f"synthetic_doc_{i:03d}.jpg")

        # Photos: colorful varied patterns
        photo_arr = np.random.randint(0, 180, (224, 224, 3), dtype=np.uint8)
        photo_img = Image.fromarray(photo_arr)
        photo_img.save(photos_dir / f"synthetic_photo_{i:03d}.jpg")

    print("Finished generating synthetic dataset.")


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for testing.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "dataset" / "synthetic"),
        help="Target output directory for synthetic images.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Number of images per class to generate.",
    )
    args = parser.parse_args()
    generate_dummy_data(Path(args.output_dir), args.count)


if __name__ == "__main__":
    main()
