/**
 * Smart Image & Document Detector - Browser Inference Client
 * 
 * Classifies an image as either 'document' (class 0) or 'photograph' (class 1)
 * using MobileNetV2 in TensorFlow.js.
 * 
 * IMPORTANT PREPROCESSING CONTRACT:
 * The model architecture includes an internal Rescaling layer:
 *   Rescaling(1./127.5, offset=-1)
 * Therefore, inputs must be provided as raw [0, 255] pixel floats.
 * DO NOT normalize with `.div(127.5).sub(1)` in JavaScript, as this
 * causes severe double-normalization errors.
 */

class SmartImageDetector {
    /**
     * @param {Object} options
     * @param {string} options.modelUrl - Path to model.json (default: '../models/tfjs_uint8/model.json')
     * @param {number} options.threshold - Classification threshold (default: 0.5)
     */
    constructor(options = {}) {
        this.modelUrl = options.modelUrl || '../models/tfjs_uint8/model.json';
        this.threshold = options.threshold !== undefined ? options.threshold : 0.5;
        this.model = null;
        this.isLoading = false;
    }

    /**
     * Load the TensorFlow.js GraphModel.
     * @returns {Promise<boolean>}
     */
    async load(customUrl = null) {
        if (customUrl) {
            this.modelUrl = customUrl;
        }

        if (this.model) {
            this.model.dispose();
            this.model = null;
        }

        this.isLoading = true;
        try {
            console.log(`[SmartDetector] Loading GraphModel from ${this.modelUrl}...`);
            this.model = await tf.loadGraphModel(this.modelUrl);
            this.isLoading = false;
            console.log('[SmartDetector] Model loaded successfully.');
            return true;
        } catch (error) {
            this.isLoading = false;
            console.error('[SmartDetector] Failed to load model:', error);
            throw error;
        }
    }

    /**
     * Classify an HTMLImageElement, HTMLCanvasElement, or ImageData.
     * 
     * @param {HTMLImageElement|HTMLCanvasElement|ImageData} imageElement
     * @returns {Promise<{class: string, label: string, probability: number, confidence: number, latencyMs: number}>}
     */
    async classify(imageElement) {
        if (!this.model) {
            await this.load();
        }

        const startTime = performance.now();

        // 1. Convert pixel element into tensor: [1, 224, 224, 3] in raw [0, 255] float range
        const inputTensor = tf.tidy(() => {
            return tf.browser.fromPixels(imageElement)
                .resizeBilinear([224, 224])
                .toFloat()
                .expandDims(0); // [0, 255] raw pixel floats
        });

        try {
            // 2. Run inference
            const predictionTensor = await this.model.predict(inputTensor);
            const predictionData = await predictionTensor.data();
            const probability = predictionData[0]; // Sigmoid score: 0.0 (Doc) -> 1.0 (Photo)

            // Clean up memory
            inputTensor.dispose();
            predictionTensor.dispose();

            const latencyMs = Math.round((performance.now() - startTime) * 10) / 10;
            const isPhoto = probability > this.threshold;
            const confidence = isPhoto ? (probability * 100) : ((1 - probability) * 100);

            return {
                class: isPhoto ? 'photo' : 'document',
                label: isPhoto ? 'Photograph' : 'Document',
                probability: parseFloat(probability.toFixed(4)),
                confidence: parseFloat(confidence.toFixed(1)),
                latencyMs: latencyMs,
                threshold: this.threshold
            };
        } catch (error) {
            inputTensor.dispose();
            console.error('[SmartDetector] Prediction failed:', error);
            throw error;
        }
    }

    /**
     * Dispose of allocated GPU memory and model weights.
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
    }
}

// Export for ES modules and standard browser globals
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SmartImageDetector;
} else if (typeof window !== 'undefined') {
    window.SmartImageDetector = SmartImageDetector;
}
