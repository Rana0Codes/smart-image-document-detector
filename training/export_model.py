"""
Export Model Pipeline for Smart Image & Document Detector.

Rebuilds the MobileNetV2 graph with the embedded Rescaling layer, loads saved
weights from checkpoint, and exports both the TensorFlow SavedModel and
TensorFlow.js Float32 formats.
"""

import argparse
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Rescaling
from tensorflow.keras.models import Model
import tensorflowjs as tfjs


def export(checkpoint_path: Path, saved_model_dir: Path, tfjs_dir: Path):
    print("=" * 60)
    print("Smart Image & Document Detector - Model Export")
    print(f"Checkpoint:   {checkpoint_path.resolve()}")
    print(f"SavedModel:   {saved_model_dir.resolve()}")
    print(f"TFJS Float32: {tfjs_dir.resolve()}")
    print("=" * 60)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Rebuild graph with embedded rescaling
    print("\nRebuilding model architecture...")
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = Input(shape=(224, 224, 3), name="inputs")
    # Embedded rescaling ensures input contract is raw [0, 255]
    x = Rescaling(1.0 / 127.5, offset=-1.0, name="rescaling")(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dropout(0.2, name="dropout")(x)
    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    model = Model(inputs=inputs, outputs=predictions, name="smart_detector")

    print(f"Loading weights from {checkpoint_path}...")
    model.load_weights(str(checkpoint_path))

    # Export SavedModel
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving TensorFlow SavedModel to: {saved_model_dir}...")
    tf.saved_model.save(model, str(saved_model_dir))
    print("SavedModel exported successfully.")

    # Export Float32 TensorFlow.js
    tfjs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nConverting to TensorFlow.js Float32 at: {tfjs_dir}...")
    tfjs.converters.convert_tf_saved_model(
        saved_model_dir=str(saved_model_dir),
        output_dir=str(tfjs_dir),
    )
    print("TensorFlow.js Float32 export complete!")


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Export Smart Image & Document Detector model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(project_root / "training" / "checkpoints" / "cp.weights.h5"),
        help="Path to trained weights checkpoint.",
    )
    parser.add_argument(
        "--saved-model-dir",
        type=str,
        default=str(project_root / "models" / "tensorflow_savedmodel"),
        help="Destination directory for SavedModel.",
    )
    parser.add_argument(
        "--tfjs-dir",
        type=str,
        default=str(project_root / "models" / "tfjs_float32"),
        help="Destination directory for TensorFlow.js Float32 model.",
    )
    args = parser.parse_args()

    export(
        checkpoint_path=Path(args.checkpoint),
        saved_model_dir=Path(args.saved_model_dir),
        tfjs_dir=Path(args.tfjs_dir),
    )


if __name__ == "__main__":
    main()
