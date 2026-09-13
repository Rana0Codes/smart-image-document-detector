# Trained Model Artifacts (`models/`)

This directory contains the trained weights and exported models for **Smart Image & Document Detector** across formats.

## Model Formats

| Directory | Format | Precision | File Size | Recommended Target |
|---|---|---|---|---|
| [`tfjs_uint8/`](tfjs_uint8/) | TensorFlow.js GraphModel | UINT8 Quantized | **~2.2 MB** (1 shard) | **Web Browsers (Recommended)** |
| [`tfjs_float32/`](tfjs_float32/) | TensorFlow.js GraphModel | Float32 | **~8.8 MB** (3 shards) | Web / Node.js High Precision |
| [`tensorflow_savedmodel/`](tensorflow_savedmodel/) | TensorFlow SavedModel | Float32 | **~1.5 MB** (`.pb`) + variables | Python Serving / FastAPI / Cloud |

---

## Preprocessing Contract & Input Shape

All three models share the exact same input specification:
- **Input Tensor**: `[batch_size, 224, 224, 3]`
- **Color Format**: RGB
- **Input Value Range**: **Raw $[0, 255]$ pixel floats**
- **Embedded Rescaling**: The computational graph embeds `tf.keras.layers.Rescaling(1./127.5, offset=-1)` as its first layer.

> [!CAUTION]
> **Do not normalize inputs to $[-1, 1]$ before passing to the model!**
> Because the model contains internal rescaling, passing pre-normalized $[-1, 1]$ values causes double-normalization, collapsing document predictions into photographs.

---

## Output Semantics & Decision Threshold

The model outputs a single scalar probability $p \in [0.0, 1.0]$ via a final `Dense(1, activation='sigmoid')` layer:
- $p < 0.50 \rightarrow$ **Document (Class 0)**
- $p \ge 0.50 \rightarrow$ **Photograph (Class 1)**

Confidence calculation:
- If $p < 0.50$: $\text{Confidence} = (1 - p) \times 100\%$
- If $p \ge 0.50$: $\text{Confidence} = p \times 100\%$

---

## Cryptographic Hashes (Integrity Verification)

### `models/tensorflow_savedmodel/`
- `saved_model.pb`: `F3FD7556C3ADFCF7F2B274D0FCBC14C1E5C2BE7D0EDE20E32E163D61189B7EA3`
- `fingerprint.pb`: `F203994BC5FC58A888EFC85F0A0CF4FD6335E401034BE2CBE02C43409DB7FF91`
- `variables/variables.data-00000-of-00001`: `96ACA441A8602387DC39FF7DB553D946B3BAACE77C3FE314625F66BC5EC9BE37`
- `variables/variables.index`: `E6525EBC4120792ECA24D35EE0F1439D472218E374E6D0B9C8F31D44CD9A9A05`

### `models/tfjs_uint8/`
- `model.json`: `B56F465335EE115FA1EE5BC942470861A805E7D51D12B349EBC73F13F08AD122`
- `group1-shard1of1.bin`: `8F1E3D2F8FE89A22AFB22EC388FFEDF7B4EA6217278A5C6C299706F2FD73533D`

### Checkpoint (`training/checkpoints/`)
- `cp.weights.h5`: `B4C90515101DA7BC5EBD569E410E1398557D9344AF20BAAC6D83CC3038F71360`
