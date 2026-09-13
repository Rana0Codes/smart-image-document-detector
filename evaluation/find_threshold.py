"""
Threshold Optimization Sweep for Smart Image & Document Detector.

Sweeps classification thresholds from 0.01 to 0.99 to determine the optimal
decision boundary balancing document and photograph recall.
"""

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def find_optimal_threshold(model_path: Path, dataset_path: Path, count: int = 200):
    print("=" * 60)
    print("Smart Image & Document Detector - Decision Threshold Sweep")
    print(f"Model:   {model_path.resolve()}")
    print(f"Dataset: {dataset_path.resolve()}")
    print(f"Samples: {count}")
    print("=" * 60)

    test_ds = image_dataset_from_directory(
        str(dataset_path),
        class_names=["document", "photo"],
        validation_split=0.1,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=1,
    )

    model = tf.saved_model.load(str(model_path))
    infer = model.signatures["serving_default"]

    y_true = []
    y_pred_probs = []

    print(f"Collecting predictions for {count} samples...")
    for img, label in test_ds.take(count):
        img_val = img.numpy()
        true_label = label.numpy()[0]

        output_dict = infer(tf.constant(img_val))
        output_key = list(output_dict.keys())[0]
        prob = float(output_dict[output_key].numpy()[0][0])

        y_true.append(true_label)
        y_pred_probs.append(prob)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    best_threshold = 0.5
    best_acc = 0.0
    best_doc_acc = 0.0
    best_photo_acc = 0.0

    print("\n--- Sweeping Thresholds (0.05 to 0.95) ---")
    for thresh in np.arange(0.05, 0.96, 0.05):
        preds = (y_pred_probs > thresh).astype(int)
        correct = preds == y_true
        overall_acc = np.mean(correct)

        doc_mask = y_true == 0
        photo_mask = y_true == 1

        doc_acc = np.mean(correct[doc_mask]) if np.sum(doc_mask) > 0 else 0
        photo_acc = np.mean(correct[photo_mask]) if np.sum(photo_mask) > 0 else 0

        print(
            f"  Threshold {thresh:.2f} -> Overall: {overall_acc*100:5.1f}%, "
            f"Doc Acc: {doc_acc*100:5.1f}%, Photo Acc: {photo_acc*100:5.1f}%"
        )

        if overall_acc > best_acc and doc_acc >= 0.85 and photo_acc >= 0.85:
            best_acc = overall_acc
            best_threshold = thresh
            best_doc_acc = doc_acc
            best_photo_acc = photo_acc

    print("\n" + "=" * 60)
    print(f"Optimal Threshold:   {best_threshold:.2f}")
    print(f"Overall Accuracy:    {best_acc*100:.1f}%")
    print(f"Document Accuracy:   {best_doc_acc*100:.1f}%")
    print(f"Photograph Accuracy: {best_photo_acc*100:.1f}%")
    print("=" * 60)
    return best_threshold


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Find optimal classification threshold.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(project_root / "models" / "tensorflow_savedmodel"),
        help="Path to SavedModel.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(project_root / "dataset"),
        help="Path to dataset directory.",
    )
    parser.add_argument("--count", type=int, default=200, help="Number of samples to evaluate.")
    args = parser.parse_args()

    find_optimal_threshold(Path(args.model_dir), Path(args.data_dir), args.count)


if __name__ == "__main__":
    main()
