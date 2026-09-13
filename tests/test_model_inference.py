"""
Automated Regression and Inference Test Suite for Smart Image & Document Detector.

Tests:
1. SavedModel artifact loading and signature verification.
2. Input range contract: [0, 255] raw pixel ingestion.
3. Prediction accuracy on sample documents and photos.
4. Regression guard: Catches accidental double-normalization.
"""

from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import tensorflow as tf


class TestSmartDetectorInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.model_dir = cls.project_root / "models" / "tensorflow_savedmodel"
        cls.samples_dir = cls.project_root / "dataset" / "samples"

        # Ensure model exists
        if not cls.model_dir.exists():
            raise FileNotFoundError(f"Model directory missing: {cls.model_dir}")

        cls.model = tf.saved_model.load(str(cls.model_dir))
        cls.infer = cls.model.signatures["serving_default"]

    def _load_and_preprocess(self, image_path: Path) -> np.ndarray:
        """Load image as [1, 224, 224, 3] float32 array in [0, 255]."""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            return np.expand_dims(arr, axis=0)

    def test_01_model_loading_and_signature(self):
        """Verify model loads and serving_default signature exists."""
        self.assertIsNotNone(self.infer)
        self.assertIn("inputs", self.infer.structured_input_signature[1])

    def test_02_document_sample_inference(self):
        """Verify document samples are classified as Document (< 0.5)."""
        doc_dir = self.samples_dir / "document"
        doc_files = list(doc_dir.glob("*.jpg")) + list(doc_dir.glob("*.png"))
        self.assertGreater(len(doc_files), 0, "No sample document images found")

        doc_correct = 0
        for f in doc_files:
            img_tensor = self._load_and_preprocess(f)
            self.assertGreater(img_tensor.max(), 1.0, "Input image must be in [0, 255]")

            out = self.infer(tf.constant(img_tensor))
            prob = float(list(out.values())[0].numpy()[0][0])
            if prob < 0.5:
                doc_correct += 1

        accuracy = doc_correct / len(doc_files)
        self.assertGreaterEqual(
            accuracy,
            0.80,
            f"Document accuracy ({accuracy*100:.1f}%) fell below 80% threshold",
        )

    def test_03_photo_sample_inference(self):
        """Verify photo samples are classified as Photograph (> 0.5)."""
        photo_dir = self.samples_dir / "photo"
        photo_files = (
            list(photo_dir.glob("*.jpg"))
            + list(photo_dir.glob("*.png"))
            + list(photo_dir.glob("*.webp"))
        )
        self.assertGreater(len(photo_files), 0, "No sample photo images found")

        photo_correct = 0
        for f in photo_files:
            img_tensor = self._load_and_preprocess(f)
            out = self.infer(tf.constant(img_tensor))
            prob = float(list(out.values())[0].numpy()[0][0])
            if prob > 0.5:
                photo_correct += 1

        accuracy = photo_correct / len(photo_files)
        self.assertGreaterEqual(
            accuracy,
            0.80,
            f"Photo accuracy ({accuracy*100:.1f}%) fell below 80% threshold",
        )

    def test_04_double_normalization_regression_guard(self):
        """
        Regression Guard:
        Verifies that double-normalizing inputs [-1, 1] into a model that already
        has an internal Rescaling layer causes a massive distortion in document output.
        This test documents and ensures our preprocessing contracts remain intact.
        """
        doc_dir = self.samples_dir / "document"
        target_doc = doc_dir / "doc_10000.jpg"
        if not target_doc.exists():
            target_doc = next(doc_dir.glob("*.jpg"))

        raw_pixels = self._load_and_preprocess(target_doc)

        # Raw input -> Should be strong document (prob < 0.1)
        out_raw = self.infer(tf.constant(raw_pixels))
        prob_raw = float(list(out_raw.values())[0].numpy()[0][0])
        self.assertLess(
            prob_raw, 0.1, f"Raw pixel input should yield strong document score, got {prob_raw}"
        )

        # Double normalized input -> Shifts to photo (> 0.5)
        double_norm = (raw_pixels / 127.5) - 1.0
        out_dn = self.infer(tf.constant(double_norm))
        prob_dn = float(list(out_dn.values())[0].numpy()[0][0])
        self.assertGreater(
            prob_dn,
            0.5,
            f"Double normalization should erroneously shift document to photo (>0.5), got {prob_dn}",
        )


if __name__ == "__main__":
    unittest.main()
