"""
Download Sample Images for Quick Testing.

Downloads a small collection of CC0 / Wikimedia Commons documents and photographs
for zero-configuration smoke testing and development.
"""

import argparse
import os
from pathlib import Path
import requests

DOC_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Receipt_for_a_Purchase_at_a_Grocery_Store.jpg/480px-Receipt_for_a_Purchase_at_a_Grocery_Store.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Lorem_ipsum_design.svg/480px-Lorem_ipsum_design.svg.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Tax_return_form_1040_2011.png/480px-Tax_return_form_1040_2011.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Invoice.svg/480px-Invoice.svg.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/US_Passport_Card_Front.jpg/480px-US_Passport_Card_Front.jpg",
]

PHOTO_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/480px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/480px-Cat_November_2010-1a.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/480px-Golde33443.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Hauskatze_langhaar.jpg/480px-Hauskatze_langhaar.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Dog_in_a_bucket.jpg/480px-Dog_in_a_bucket.jpg",
]


def download_batch(urls, target_dir: Path, prefix: str):
    target_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "SmartDetector-Downloader/1.0 (MachineLearningResearch)"}
    count = 0
    for i, url in enumerate(urls):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                file_path = target_dir / f"{prefix}_{count}.jpg"
                with open(file_path, "wb") as f:
                    f.write(res.content)
                count += 1
                print(f"Downloaded {prefix}_{count}.jpg")
            else:
                print(f"HTTP {res.status_code} for {url}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    print(f"Completed downloading {count} images to {target_dir}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Download sample evaluation images.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "dataset" / "samples"),
        help="Destination directory for downloaded samples.",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    docs_dir = base_dir / "document"
    photos_dir = base_dir / "photo"

    print("Downloading sample documents...")
    download_batch(DOC_URLS, docs_dir, "sample_doc")

    print("\nDownloading sample photographs...")
    download_batch(PHOTO_URLS, photos_dir, "sample_photo")


if __name__ == "__main__":
    main()
