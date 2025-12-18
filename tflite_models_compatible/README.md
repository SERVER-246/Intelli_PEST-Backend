# TFLite Compatible Models

This folder contains TFLite models converted with **maximum backward compatibility** for Android TFLite Runtime 2.14.0 and above.

## Why This Folder Exists

The models in `tflite_models/` were converted with TensorFlow 2.20, which produces TFLite models using newer operation versions (e.g., `FULLY_CONNECTED version 12`). These don't work with older TFLite runtimes.

This folder contains models converted with:
- **ONNX opset 11** (older, more compatible)
- **Legacy TFLite converter settings** (no experimental features)
- **Standard TFLite builtins only** (no flex delegate needed)

## Conversion Scripts

### Option 1: `convert_compatible.py` (Recommended)
Uses float16 quantization with compatibility settings.

```bash
cd tflite_models_compatible
python convert_compatible.py
```

### Option 2: `convert_legacy.py` (Maximum Compatibility)
Uses strictest backward compatibility settings. Use this if Option 1 doesn't work.

```bash
cd tflite_models_compatible
python convert_legacy.py
```

## Model Input Dimensions

**IMPORTANT**: Different models expect different input sizes!

| Model | Input Size | Notes |
|-------|------------|-------|
| mobilenet_v2 | 224×224 | Standard ImageNet |
| resnet50 | 224×224 | Standard ImageNet |
| inception_v3 | **299×299** | InceptionV3 specific |
| efficientnet_b0 | 224×224 | NOT 256×256! |
| darknet53 | 224×224 | Standard |
| alexnet | 224×224 | Standard |
| yolo11n-cls | 224×224 | Standard |
| ensemble_attention | 224×224 | Standard |
| ensemble_concat | 224×224 | Standard |
| ensemble_cross | 224×224 | Standard |
| super_ensemble | 224×224 | Standard |

## Android Integration

### 1. Copy Models to Android Project
```
app/src/main/assets/models/
├── mobilenet_v2.tflite
├── resnet50.tflite
├── inception_v3.tflite
└── ...
```

### 2. Read Input Size from Metadata
```kotlin
// Load model metadata to get correct input size
val metadata = loadJsonFromAssets("models/model_metadata.json")
val inputSize = metadata.models[modelName].input_size  // 224 or 299
```

### 3. Resize Image to Correct Size
```kotlin
fun preprocessImage(bitmap: Bitmap, modelName: String): TensorBuffer {
    val inputSize = getModelInputSize(modelName)  // 224 or 299
    val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
    // ... rest of preprocessing
}
```

### 4. TFLite Interpreter Setup
```kotlin
val options = Interpreter.Options()
    .setNumThreads(4)
    .setUseNNAPI(false)  // Disable NNAPI for compatibility

val interpreter = Interpreter(loadModelFile(modelPath), options)
```

## Compatibility Matrix

| TFLite Runtime | tflite_models/ | tflite_models_compatible/ |
|----------------|----------------|---------------------------|
| 2.14.0 | ❌ | ✅ |
| 2.15.0 | ❌ | ✅ |
| 2.16.0 | ❌ | ✅ |
| 2.16.1 | ⚠️ | ✅ |
| 2.17.0+ | ✅ | ✅ |

## Troubleshooting

### Error: `Didn't find op for builtin opcode 'FULLY_CONNECTED' version '12'`
- You're using models from `tflite_models/` with an old TFLite runtime
- Solution: Use models from this folder (`tflite_models_compatible/`)

### Error: `Got invalid dimensions for input`
- Wrong input size for the model
- Check `model_metadata.json` for correct input dimensions
- InceptionV3 needs 299×299, others need 224×224

### Error: `Cannot create interpreter`
- Model file may be corrupted
- Re-run conversion script
- Check if TFLite dependency is correctly included in build.gradle

## Files

```
tflite_models_compatible/
├── README.md                  # This file
├── convert_compatible.py      # Main conversion script (float16)
├── convert_legacy.py          # Legacy conversion script (max compat)
├── model_metadata.json        # Generated metadata with input sizes
├── mobilenet_v2.tflite        # Converted models...
├── resnet50.tflite
├── inception_v3.tflite
├── efficientnet_b0.tflite
├── darknet53.tflite
├── alexnet.tflite
├── yolo11n-cls.tflite
├── ensemble_attention.tflite
├── ensemble_concat.tflite
├── ensemble_cross.tflite
└── super_ensemble.tflite
```

## Generation Command

To regenerate all compatible models:

```bash
# Activate virtual environment
cd D:\Intelli_PEST-Backend
.\venv_tflite\Scripts\activate

# Run conversion
cd tflite_models_compatible
python convert_legacy.py  # or convert_compatible.py
```

## Model Metadata JSON Format

```json
{
  "generated_at": "2025-12-18T...",
  "target_runtime": "TFLite 2.14.0+",
  "models": {
    "mobilenet_v2": {
      "status": "success",
      "input_size": 224,
      "file_size_mb": 3.5,
      "tflite_path": "..."
    }
  },
  "input_format": {
    "tensor_format": "NHWC",
    "data_type": "float32",
    "normalization": "divide by 255.0 (0-1 range)",
    "channel_order": "RGB"
  }
}
```

Use this metadata in your Android app to dynamically get the correct input size for each model.
