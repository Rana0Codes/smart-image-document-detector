# Deployment Guide (`docs/DEPLOYMENT.md`)

This guide explains how to deploy **Smart Image & Document Detector** across Web Browsers, Node.js backends, and Python microservices.

---

## 1. Browser Deployment (TensorFlow.js) - Recommended

### A. Including Dependencies
Include TensorFlow.js in your HTML:
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
<script src="smart-image-detector.js"></script>
```

### B. Instantiating & Classifying
```javascript
// 1. Initialize detector pointing to the UINT8 model
const detector = new SmartImageDetector({
    modelUrl: './models/tfjs_uint8/model.json',
    threshold: 0.5
});

// 2. Pre-load weights (asynchronous)
await detector.load();

// 3. Classify an HTML image element, canvas, or video frame
const imgElement = document.getElementById('my-uploaded-image');
const result = await detector.classify(imgElement);

console.log(result);
/* Output:
{
    class: 'document',
    label: 'Document',
    probability: 0.0034,
    confidence: 99.7,
    latencyMs: 16.2,
    threshold: 0.5
}
*/

// 4. Clean up GPU memory when done
detector.dispose();
```

---

## 2. Node.js Deployment (`@tensorflow/tfjs-node`)

Install the native TensorFlow.js Node binding:
```bash
npm install @tensorflow/tfjs-node
```

```javascript
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

async function runNodeInference(imagePath) {
    const model = await tf.loadGraphModel('file://./models/tfjs_uint8/model.json');

    // Read and decode image buffer
    const imageBuffer = fs.readFileSync(imagePath);
    const tensor = tf.node.decodeImage(imageBuffer, 3)
        .resizeBilinear([224, 224])
        .toFloat()
        .expandDims(0); // [0, 255] raw pixels

    const prediction = await model.predict(tensor).data();
    const prob = prediction[0];

    tensor.dispose();
    model.dispose();

    return prob > 0.5 ? 'photo' : 'document';
}
```

---

## 3. Python FastAPI / Flask Deployment

```python
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Smart Image & Document Detector API")

MODEL_PATH = Path("models/tensorflow_savedmodel")
model = tf.saved_model.load(str(MODEL_PATH))
infer = model.signatures["serving_default"]

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Open and format image
    img = Image.open(file.file).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0) # [0, 255]

    # Run inference
    res = infer(tf.constant(arr))
    prob = float(list(res.values())[0].numpy()[0][0])

    is_photo = prob > 0.5
    return {
        "class": "photo" if is_photo else "document",
        "label": "Photograph" if is_photo else "Document",
        "probability": round(prob, 4),
        "confidence": round(prob * 100 if is_photo else (1 - prob) * 100, 1)
    }
```

---

## Memory Management Best Practices

In browser environments:
1. Wrap tensor operations in `tf.tidy()` or explicitly call `.dispose()` on all created tensors to avoid WebGL memory leaks.
2. Call `detector.dispose()` if destroying the view or component.
3. Use the `UINT8` model for 75% faster asset downloads over the network.
