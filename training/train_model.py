"""
Training Pipeline for Smart Image & Document Detector.
 
Architecture:
- Backbone: MobileNetV2 (ImageNet pretrained, frozen)
- Input: 224x224x3 RGB image in [0, 255]
- Rescaling: tf.keras.layers.Rescaling(1./127.5, offset=-1) -> [-1, 1]
- Head: GlobalAveragePooling2D -> Dropout(0.2) -> Dense(1, activation='sigmoid')
- Output: Binary probability (0: Document, 1: Photograph)
"""

import argparse
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory


def train(
    data_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    img_size: tuple = (224, 224),
    learning_rate: float = 0.001,
):
    print("=" * 60)
    print("Smart Image & Document Detector - Training Pipeline")
    print(f"Dataset:     {data_dir.resolve()}")
    print(f"Output:      {output_dir.resolve()}")
    print(f"Checkpoints: {checkpoint_dir.resolve()}")
    print(f"Epochs:      {epochs}, Batch size: {batch_size}, Image size: {img_size}")
    print("=" * 60)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    # Load dataset with 80/20 train/validation split
    print("\nLoading training dataset...")
    train_ds = image_dataset_from_directory(
        str(data_dir),
        class_names=["document", "photo"],
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

    print("Loading validation dataset...")
    val_ds = image_dataset_from_directory(
        str(data_dir),
        class_names=["document", "photo"],
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    print(f"\nClass mapping: {class_names}")
    print(f"  0 -> '{class_names[0]}'")
    print(f"  1 -> '{class_names[1]}'")

    # Build model architecture
    print("\nConstructing MobileNetV2 model graph...")
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = Input(shape=(224, 224, 3), name="inputs")
    # CRITICAL: Embed rescaling directly inside the graph so external callers can feed raw [0, 255] pixels
    x = Rescaling(1.0 / 127.5, offset=-1.0, name="rescaling")(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dropout(0.2, name="dropout")(x)
    predictions = Dense(1, activation="sigmoid", name="predictions")(x)

    model = Model(inputs=inputs, outputs=predictions, name="smart_detector")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Checkpoint configuration
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "cp.weights.h5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        save_weights_only=True,
        verbose=1,
    )

    if checkpoint_path.exists():
        print(f"\nFound existing checkpoint at {checkpoint_path}. Loading weights...")
        model.load_weights(str(checkpoint_path))

    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[cp_callback],
    )

    # Save final model as TensorFlow SavedModel
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final model as SavedModel to: {output_dir}")
    tf.saved_model.save(model, str(output_dir))
    print("Training and export complete!")
    return model, history


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train Smart Image & Document Detector model.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(project_root / "dataset"),
        help="Path to dataset containing 'document' and 'photo' subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "models" / "tensorflow_savedmodel"),
        help="Path to export final SavedModel.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(project_root / "training" / "checkpoints"),
        help="Path to save training checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    args = parser.parse_args()

    train(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
