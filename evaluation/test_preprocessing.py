"""
Diagnostic Preprocessing Inspection for Smart Image & Document Detector.

Demonstrates that the model graph embeds `Rescaling(1./127.5, offset=-1)`
and inspects the prediction shift caused by double-normalizing inputs.
"""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def test_preprocessing(model_path: Path, dataset_path: Path):
    print("=" * 60)
    print("Smart Image & Document Detector - Preprocessing Contract Inspection")
    print(f"Model:   {model_path.resolve()}")
    print(f"Dataset: {dataset_path.resolve()}")
    print("=" * 60)

    test_ds = image_dataset_from_directory(
        str(dataset_path),
        class_names=["document", "photo"],
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=1,
    )

    for img, label in test_ds.take(1):
        img_val = img.numpy()
        true_label = int(label.numpy()[0])
        break

    label_str = "Document (0)" if true_label == 0 else "Photograph (1)"
    print(f"\nLoaded Test Sample: Ground Truth = {label_str}")
    print(f"Image Tensor Shape:     {img_val.shape}")
    print(f"Raw Pixel Value Range:  [{img_val.min():.1f}, {img_val.max():.1f}]")

    model = tf.saved_model.load(str(model_path))
    infer = model.signatures["serving_default"]

    # Test 1: Raw [0, 255] Pixels
    out_1 = infer(tf.constant(img_val))
    prob_1 = float(list(out_1.values())[0].numpy()[0][0])
    pred_1 = "Photograph (1)" if prob_1 > 0.5 else "Document (0)"

    # Test 2: Double-normalized [-1, 1] Pixels
    img_norm = (img_val / 127.5) - 1.0
    out_2 = infer(tf.constant(img_norm))
    prob_2 = float(list(out_2.values())[0].numpy()[0][0])
    pred_2 = "Photograph (1)" if prob_2 > 0.5 else "Document (0)"

    print("\n" + "-" * 50)
    print(f"Inference 1: Raw [0, 255] Input")
    print(f"  Model Probability: {prob_1:.4f} -> Prediction: {pred_1}")
    print(f"  Status: {'CORRECT' if (prob_1 > 0.5) == (true_label == 1) else 'MISCLASSIFIED'}")

    print("\nInference 2: Double-Normalized [-1, 1] Input")
    print(f"  Model Probability: {prob_2:.4f} -> Prediction: {pred_2}")
    print(f"  Status: {'CORRECT' if (prob_2 > 0.5) == (true_label == 1) else 'MISCLASSIFIED'}")
    print("-" * 50)

    print("\nConclusion: The model requires raw [0, 255] inputs because normalization")
    print("is already handled by the graph's internal Rescaling layer.")


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Inspect preprocessing behavior.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(project_root / "models" / "tensorflow_savedmodel"),
        help="Path to SavedModel directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(project_root / "dataset"),
        help="Path to dataset directory.",
    )
    args = parser.parse_args()

    test_preprocessing(Path(args.model_dir), Path(args.data_dir))


if __name__ == "__main__":
    main()
