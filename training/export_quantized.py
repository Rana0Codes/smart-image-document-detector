"""
Export Quantized TensorFlow.js Model (UINT8).

Converts the TensorFlow SavedModel into an 8-bit quantized TensorFlow.js graph
model (~2.2 MB), achieving ~75% file size compression for web browser deployment.
"""

import argparse
from pathlib import Path
import tensorflowjs as tfjs


def export_quantized(saved_model_dir: Path, output_dir: Path):
    print("=" * 60)
    print("Smart Image & Document Detector - UINT8 Quantized TFJS Export")
    print(f"Source SavedModel: {saved_model_dir.resolve()}")
    print(f"Target Output:     {output_dir.resolve()}")
    print("=" * 60)

    if not saved_model_dir.exists():
        raise FileNotFoundError(f"SavedModel not found at: {saved_model_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nExecuting UINT8 quantization conversion...")
    tfjs.converters.convert_tf_saved_model(
        saved_model_dir=str(saved_model_dir),
        output_dir=str(output_dir),
        quantization_dtype_map={"uint8": "*"},
    )
    print("\nConversion successful! Quantized model ready in:", output_dir)


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Export quantized UINT8 TFJS model.")
    parser.add_argument(
        "--saved-model-dir",
        type=str,
        default=str(project_root / "models" / "tensorflow_savedmodel"),
        help="Path to source SavedModel.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "models" / "tfjs_uint8"),
        help="Destination directory for UINT8 quantized TFJS model.",
    )
    args = parser.parse_args()

    export_quantized(Path(args.saved_model_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
