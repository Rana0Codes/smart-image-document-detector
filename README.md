# Smart Image & Document Detector: Client-Side Document vs. Photograph AI Classifier

[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-v4.10.0-orange.svg)](https://js.tensorflow.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.15+-blue.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture: MobileNetV2](https://img.shields.io/badge/Backbone-MobileNetV2-purple.svg)](https://arxiv.org/abs/1801.04381)
[![Model Size](https://img.shields.io/badge/Model%20Size-2.2%20MB%20(UINT8)-brightgreen.svg)](models/tfjs_uint8/)

**Smart Image & Document Detector** is an open-source, client-side machine learning classifier built on top of **MobileNetV2**. It classifies images in real time directly inside the web browser as either a **Document** (e.g., invoices, receipts, legal forms, tax documents) or a **Photograph** (e.g., portraits, landscapes, wildlife, street scenes).

Originally designed to automate adaptive image processing and web ingestion pipelines, this model enables web applications to dynamically toggle optimization strategies—applying aggressive lossy compression (WebP / AVIF) to photographs while preserving text clarity with lossless or high-fidelity compression for documents and forms.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Interactive GUI Demo](#interactive-gui-demo)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Classes & Output Semantics](#classes--output-semantics)
- [Dataset](#dataset)
- [Model Performance & Benchmarks](#model-performance--benchmarks)
- [Installation](#installation)
- [Running the Browser Demo](#running-the-browser-demo)
- [Running Evaluation](#running-evaluation)
- [Training the Model](#training-the-model)
- [Adding More Dataset Images](#adding-more-dataset-images)
- [Exporting TensorFlow.js Models](#exporting-tensorflowjs-models)
- [Browser Integration Example](#browser-integration-example)
- [Critical Preprocessing Requirements](#critical-preprocessing-requirements)
- [Model Files & Checksums](#model-files--checksums)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Real-World Limitations](#real-world-limitations)
- [Privacy & Client-Side Execution](#privacy--client-side-execution)
- [License & Third-Party Attribution](#license--third-party-attribution)
- [Contributing](#contributing)

---

## What It Does
 
In automated document processing, scanning, and web upload workflows, images require vastly different treatment:
- **Photographs**: Can be aggressively compressed (e.g., lower-quality WebP/JPEG) for significant bandwidth and storage savings without noticeable visual loss.
- **Documents**: Contain text, fine lines, stamps, and barcodes. Lossy compression introduces artifacts that render text blurry and illegible.

Smart Image & Document Detector automatically identifies the image type in the browser in under 30 milliseconds, eliminating manual dropdowns and protecting document legibility.

---

## Interactive GUI Demo

A self-contained, responsive web demo is included in the [`demo/`](demo/) directory.

```
┌─────────────────────────────────────────────────────────────┐
│                    Smart Image Detector                     │
│                                                             │
│   Drop an image here or select a file                       │
│                                                             │
│                    [ Choose Image ]                         │
│                                                             │
│                ┌───────────────────────┐                    │
│                │                       │                    │
│                │      Image Preview    │                    │
│                │                       │                    │
│                └───────────────────────┘                    │
│                                                             │
│  Result:     DOCUMENT                                       │
│  Confidence: 99.7%                                          │
│  Latency:    18.4 ms                                        │
│  Model:      TensorFlow.js UINT8 Quantized (~2.2 MB)        │
│                                                             │
│  Recommendation: Apply text-preserving lossless compression │
└─────────────────────────────────────────────────────────────┘
```

Launch the demo locally in one command:
```bash
python -m http.server 8000
```
Then visit: **[http://localhost:8000/demo/](http://localhost:8000/demo/)**

---

## Key Features

- **100% Client-Side Privacy**: Runs locally in the browser via WebGL/WASM through TensorFlow.js. User files are never uploaded to a server.
- **Ultralight Deployment**: Quantized **UINT8 model is just ~2.2 MB**, downloading in under 1 second on typical connections.
- **Fast Inference**: Sub-30ms classification on modern desktop and mobile browsers.
- **Embedded Rescaling**: Model graph embeds `tf.keras.layers.Rescaling(1./127.5, offset=-1)`, accepting standard raw $[0, 255]$ pixel values directly.
- **Zero-Config Testing**: Includes curated sample images in [`dataset/samples/`](dataset/samples/) for immediate offline testing.

---

## How It Works

1. **Input**: Image is resized to $224 \times 224 \times 3$ RGB.
2. **Internal Normalization**: Graph rescales raw $[0, 255]$ inputs to $[-1.0, 1.0]$.
3. **Feature Extraction**: MobileNetV2 extracts 1,280 high-level feature maps.
4. **Pooling & Regularization**: `GlobalAveragePooling2D` collapses spatial dimensions; `Dropout(0.2)` guards against overconfidence.
5. **Logistic Sigmoid**: Output neuron produces probability $p \in [0.0, 1.0]$.
6. **Decision**: $p < 0.50 \rightarrow$ **Document**; $p \ge 0.50 \rightarrow$ **Photograph**.

---

## Model Architecture

```
Layer (type)                     Output Shape          Param #
=================================================================
inputs (InputLayer)              [(None, 224, 224, 3)] 0
rescaling (Rescaling)            (None, 224, 224, 3)   0
mobilenetv2_1.00_224 (Functional)(None, 7, 7, 1280)    2,257,984 (Frozen)
global_avg_pool (GlobalAvgPool2D)(None, 1280)          0
dropout (Dropout)                (None, 1280)          0
predictions (Dense)              (None, 1)             1,281
=================================================================
Total params: 2,259,265 (8.62 MB)
Trainable params: 1,281 (5.00 KB)
Non-trainable params: 2,257,984 (8.61 MB)
```

For comprehensive mathematical and architectural details, see [`docs/MODEL.md`](docs/MODEL.md).

---

## Classes & Output Semantics

| Class Name | Target Index | Probability Range | Compression Strategy |
|---|---|---|---|
| **Document** | `0` | $p < 0.50$ | Lossless / Text-preserving (PNG, WebP lossless) |
| **Photograph** | `1` | $p \ge 0.50$ | High-efficiency lossy (WebP / JPEG quality 75) |

Confidence calculation:
$$\text{Confidence} = \begin{cases} (1 - p) \times 100\% & \text{if } p < 0.50 \\ p \times 100\% & \text{if } p \ge 0.50 \end{cases}$$

---

## Dataset

The model was trained and evaluated on **11,395 images**:
- **5,000 Documents**: Scanned paperwork, receipts, invoices, tax forms, memos, and letters from the [RVL-CDIP benchmark](https://www.cs.cmu.edu/~aharley/rvl-cdip/) (derived from UCSF Tobacco Documents).
- **6,395 Photographs**: Portraits, landscapes, wildlife, and street photography from open computer vision datasets.

### Sample Images Included in Repository
To keep `git clone` fast (< 20 MB repository footprint), the full 1.85 GB image set is excluded from Git via `.gitignore`. A curated sample set of 50 images is provided in [`dataset/samples/`](dataset/samples/) for immediate testing:
- `dataset/samples/document/` (25 sample document images)
- `dataset/samples/photo/` (25 sample photograph images)

To inspect dataset provenance and obtain the full dataset, see [`docs/DATASET.md`](docs/DATASET.md).

---

## Model Performance & Benchmarks

To ensure scientific honesty and transparency, accuracy is reported across distinct evaluation dimensions rather than claiming generalized "100% real-world accuracy":

| Evaluation Dimension | Sample Count | Scope & Dataset Source | Accuracy |
|---|---|---|---|
| **1. Controlled Held-Out Batch** | 100 images | Random 80/20 train-validation split from full 11,395 corpus | **100.0%** |
| **2. Bundled Web Demo Samples** | 12 images | 6 diverse scanned documents + 6 diverse photographs (`demo/samples/`) | **100.0%** |
| **3. Independent Real-World Documents** | 8 images | Held-out invoices, application forms, ledgers, surveys, memos | **100.0%** |
| **4. Independent Real-World Photographs** | 8 images | Portraits, city street, landscape, indoor, animals, architecture | **100.0%** |
| **5. Boundary Cases (ID Cards with Photos)**| 4 specimens | Identity cards with embedded portraits (Aadhaar / Passport style) | **Known Failure Mode** ($p > 0.90$) |

```
Controlled Test Batch Confusion Matrix (python evaluation/verify_batch.py --count 100):
                    Predicted Document    Predicted Photo
Actual Document             48                    0
Actual Photograph            0                   52
```

> [!NOTE]
> **Controlled Benchmark vs. Real-World Domain Shift**:
> 100% accuracy on standard document scans and photography does **not** imply 100% accuracy on every edge case in the wild. Identity documents containing human portrait photographs (such as Aadhaar cards, passports, driver's licenses) are classified as Photographs ($p > 0.90$) due to facial feature activations in the frozen MobileNetV2 backbone. See [`docs/REAL_WORLD_VALIDATION.md`](docs/REAL_WORLD_VALIDATION.md) for full empirical analysis and mitigation strategies.

---

## Installation

### Prerequisites
- Python 3.10 – 3.13
- Node.js 18+ (optional, for Node.js inference)

### Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/Rana0Codes/smart-image-document-detector.git
cd smart-image-document-detector

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Running the Browser Demo

Run a local HTTP server from the project root:
```bash
python -m http.server 8000
```
Open **[http://localhost:8000/demo/](http://localhost:8000/demo/)** in any modern web browser.
- Use the quick sample buttons to test receipts and photos with 1 click.
- Drag and drop your own images.
- Switch between **UINT8 Quantized** (~2.2 MB) and **Float32** (~8.8 MB) models.

---

## Running Evaluation

Run the automated validation batch benchmark:
```bash
python evaluation/verify_batch.py --count 100
```

Run threshold optimization sweep:
```bash
python evaluation/find_threshold.py --count 200
```

Run quantization footprint comparison:
```bash
python evaluation/evaluate_quantization.py
```

Run the automated unit test suite:
```bash
python -m unittest tests/test_model_inference.py
```

---

## Training the Model

To retrain the model on the full dataset:
```bash
python training/train_model.py --epochs 5 --batch-size 32
```
This will:
1. Load dataset from `dataset/` (80% train / 20% validation).
2. Save checkpoints to `training/checkpoints/cp.weights.h5`.
3. Export the SavedModel to `models/tensorflow_savedmodel/`.

See [`docs/TRAINING.md`](docs/TRAINING.md) for full training details.

---

## Adding More Dataset Images

1. Add your document images to `dataset/document/`.
2. Add your photograph images to `dataset/photo/`.
3. Validate image readability and RGB color modes:
   ```bash
   python scripts/validate_dataset.py
   ```
4. Run training:
   ```bash
   python training/train_model.py
   ```

---

## Exporting TensorFlow.js Models

### 1. Export Float32 Model (~8.8 MB)
```bash
python training/export_model.py
```

### 2. Export UINT8 Quantized Model (~2.2 MB - Recommended)
```bash
python training/export_quantized.py
```

---

## Browser Integration Example

Add Smart Image & Document Detector to any web application in 3 steps:

### 1. HTML
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
<script src="path/to/smart-image-detector.js"></script>
```

### 2. JavaScript
```javascript
// Initialize detector
const detector = new SmartImageDetector({
    modelUrl: './models/tfjs_uint8/model.json',
    threshold: 0.5
});

// Pre-load model
await detector.load();

// Classify an image element
const img = document.getElementById('user-upload');
const result = await detector.classify(img);

console.log(result);
/* Output:
{
    class: 'document',
    label: 'Document',
    probability: 0.0034,
    confidence: 99.7,
    latencyMs: 18.2,
    threshold: 0.5
}
*/
```

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for Node.js and Python FastAPI integration.

---

## Critical Preprocessing Requirements

> [!IMPORTANT]
> **Input Contract**:
> - Input shape: $224 \times 224 \times 3$ RGB
> - Expected pixel range: **Raw $[0, 255]$ values**
> - The model graph **internally rescales** $[0, 255] \rightarrow [-1, 1]$.

**Do NOT perform manual normalization in JavaScript:**
```javascript
// ❌ INCORRECT (Causes double-normalization; document accuracy drops to 0%)
tensor.div(127.5).sub(1);

// ✅ CORRECT (Pass raw [0, 255] pixels; handled by model)
tf.browser.fromPixels(imgElement).resizeBilinear([224, 224]).toFloat().expandDims(0);
```

---

## Model Files & Checksums

| File | Format | SHA256 Hash |
|---|---|---|
| `models/tfjs_uint8/model.json` | TFJS GraphModel | `B56F465335EE115FA1EE5BC942470861A805E7D51D12B349EBC73F13F08AD122` |
| `models/tfjs_uint8/group1-shard1of1.bin` | UINT8 Weights | `8F1E3D2F8FE89A22AFB22EC388FFEDF7B4EA6217278A5C6C299706F2FD73533D` |
| `models/tensorflow_savedmodel/saved_model.pb` | SavedModel Protobuf | `F3FD7556C3ADFCF7F2B274D0FCBC14C1E5C2BE7D0EDE20E32E163D61189B7EA3` |
| `training/checkpoints/cp.weights.h5` | Keras HDF5 Checkpoint | `B4C90515101DA7BC5EBD569E410E1398557D9344AF20BAAC6D83CC3038F71360` |

---

## Project Structure

```
smart-image-document-detector/
│
├── README.md                      # Primary project documentation
├── LICENSE                        # MIT License with third-party attributions
├── requirements.txt               # Minimal Python dependencies
├── .gitignore                     # Strict Git exclude rules
│
├── models/                        # Pre-trained model artifacts
│   ├── README.md                  # Model specifications & hashes
│   ├── tfjs_uint8/                # Recommended UINT8 browser model (~2.2 MB)
│   ├── tfjs_float32/              # High-precision Float32 browser model (~8.8 MB)
│   └── tensorflow_savedmodel/     # Standard TensorFlow SavedModel for Python
│
├── demo/                          # Browser GUI demonstration
│   ├── README.md                  # Demo setup & GitHub Pages instructions
│   ├── index.html                 # Interactive web demo application
│   ├── smart-image-detector.js    # Client-side inference module
│   └── samples/                   # 12 diverse test sample images
│
├── training/                      # Training & export scripts
│   ├── README.md                  # Retraining & export instructions
│   ├── train_model.py             # MobileNetV2 training pipeline
│   ├── export_model.py            # SavedModel & Float32 TFJS exporter
│   ├── export_quantized.py        # UINT8 TFJS quantization exporter
│   └── checkpoints/               # Trained model checkpoint weights
│
├── evaluation/                    # Benchmarking & diagnostics
│   ├── README.md                  # Evaluation guide & results
│   ├── verify_batch.py            # Batch accuracy & confusion matrix benchmark
│   ├── verify_real_world.py       # Real-world sample verification & domain analysis
│   ├── find_threshold.py          # Optimal threshold sweep
│   ├── test_preprocessing.py      # Preprocessing contract verification
│   └── evaluate_quantization.py   # Quantization footprint comparison
│
├── dataset/                       # Dataset directory & sample set
│   ├── README.md                  # Dataset specification & licensing
│   └── samples/                   # Curated sample images for testing
│       ├── document/              # 25 sample document images
│       └── photo/                 # 25 sample photograph images
│
├── scripts/                       # Dataset utilities
│   ├── README.md                  # Utilities overview
│   ├── validate_dataset.py        # Image integrity & RGB mode validator
│   ├── clean_dataset.py           # In-place dataset sanitization tool
│   ├── extract_and_organize.py    # Raw zip archive extractor
│   ├── download_samples.py        # Web sample downloader
│   └── generate_dummy_data.py     # Synthetic data generator for CI/CD
│
├── docs/                          # Comprehensive technical documentation
│   ├── MODEL.md                   # Deep architectural details
│   ├── TRAINING.md                # Retraining guide & hyperparameters
│   ├── DATASET.md                 # Dataset acquisition & legal analysis
│   ├── EVALUATION.md              # Detailed evaluation report
│   ├── REAL_WORLD_VALIDATION.md   # Domain shift analysis & ID card failure report
│   └── DEPLOYMENT.md              # Browser, Node.js, and Python deployment
│
└── tests/                         # Automated unit & regression test suite
    └── test_model_inference.py    # Preprocessing contract & accuracy guard
```

---

## Reproducibility

To verify full pipeline reproducibility from a clean clone:
```bash
# 1. Validate dataset samples
python scripts/validate_dataset.py --samples-only

# 2. Run automated regression test suite
python -m unittest tests/test_model_inference.py

# 3. Run batch evaluation benchmark
python evaluation/verify_batch.py --count 50

# 4. Run real-world samples validation
python evaluation/verify_real_world.py

# 5. Launch local web demo
python -m http.server 8000
```

---

## Real-World Limitations & Domain Shift

1. **Identity Cards with Embedded Face Photos**: 
   Identity documents containing portrait photographs (e.g., Aadhaar cards, passports, driver's licenses) are frequently classified as Photographs ($p > 0.85$). 
   - **Reason**: The model was trained on scanned archival monochrome documents (RVL-CDIP) vs. color photos. The frozen MobileNetV2 backbone detects human facial features and glossy plastic color textures, which overpower text features.
   - For empirical evaluation and production mitigation strategies, see [`docs/REAL_WORLD_VALIDATION.md`](docs/REAL_WORLD_VALIDATION.md).
2. **Hybrid Content**: Scanned magazine pages containing >70% photograph area and minimal text may score as Photographs ($p > 0.60$).
3. **Extreme Close-Up Crops**: A tight crop of a barcode, seal, or stamp without surrounding white paper margins may exhibit reduced confidence.
4. **Safety Default**: In production document archival pipelines, ambiguous scores ($0.35 \le p \le 0.70$) should default to lossless or visually lossless compression to prevent degradation of legal text.

---

## Privacy & Client-Side Execution

**Smart Image & Document Detector** executes entirely within the user's local browser runtime. Image pixels are processed via WebGL shaders on the client device and are discarded immediately following inference. No image content or telemetry is transmitted across the network.

---

## License & Third-Party Attribution

This project is licensed under the **MIT License** - see the [`LICENSE`](LICENSE) file for details.

### Third-Party Acknowledgements
- **MobileNetV2**: Developed by Google LLC under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
- **TensorFlow & TensorFlow.js**: Developed by Google LLC under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
- **RVL-CDIP Dataset**: Compiled by Ryerson Vision Lab from the UCSF Truth Tobacco Industry Documents archive (Public Domain litigation records).

---

## Contributing

Contributions, bug reports, and dataset augmentations are welcome! Please open an issue or pull request following the repository's coding standards.
