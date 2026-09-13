# Smart Image & Document Detector - Web Browser Demo (`demo/`)

This directory contains a static browser-based demonstration application running **TensorFlow.js**.

## Features

- **100% Client-Side Privacy**: All image inference executes natively on the client device via WebGL/WASM without sending images to any server.
- **Drag-and-Drop & File Picker**: Instant image upload, client-side decoding, and preview.
- **12 Curated Quick Samples**: 1-click testing with 6 diverse document types (invoices, legal forms, financial sheets, letterheads, questionnaires, reports) and 6 diverse photo types (portraits, nature, street, animals, architecture, indoor).
- **Real-Time Model Switching**: Toggle between the recommended **UINT8 Quantized** model (~2.2 MB) and the **Float32** model (~8.8 MB).
- **Latency & Confidence Gauges**: Real-time inference latency reporting in milliseconds, confidence percentages, and raw sigmoid probabilities.
- **Engineering Guidance**: Contextual recommendations for lossless document archiving vs lossy photographic compression.

---

## How to Run Locally

Because the demo loads TensorFlow.js models (`model.json` and `.bin` shards) via `fetch`, it must be served over an HTTP/HTTPS connection rather than the `file://` protocol.

### Option 1: Python Built-in HTTP Server (Recommended)
From the root of the repository:
```bash
python -m http.server 8000
```
Then open your browser to:
[http://localhost:8000/demo/](http://localhost:8000/demo/)

### Option 2: Node.js `serve` / `http-server`
```bash
npx serve .
```

### Option 3: VS Code Live Server
Right-click `demo/index.html` and select **"Open with Live Server"**.

---

## Deploying to GitHub Pages

The demo is a completely static site:
1. Push this repository to GitHub.
2. Navigate to **Settings** &rarr; **Pages**.
3. Under **Build and deployment**, set Source to **Deploy from a branch**, Branch to `main`, and folder to `/demo` or `/ (root)`.
4. Your demo will be live with full client-side privacy.
