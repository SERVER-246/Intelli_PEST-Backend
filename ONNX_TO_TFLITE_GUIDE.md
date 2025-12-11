# ONNX to TFLite Conversion Guide

## Current Status

✅ **ONNX Models Ready**: 11 models in `D:\Base-dir\onnx_models\`
- alexnet
- darknet53
- efficientnet_b0
- ensemble_attention
- ensemble_concat
- ensemble_cross
- inception_v3
- mobilenet_v2
- resnet50
- super_ensemble
- yolo11n-cls

## Issue

The Python environment has TensorFlow installation issues preventing direct ONNX→TFLite conversion.

##Solutions

### Option 1: Use ONNX Models Directly (RECOMMENDED)

Your Android app can use ONNX models directly with **ONNX Runtime**:

```gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'
}
```

Benefits:
- No conversion needed
- Better performance than TFLite in many cases
- Your ONNX models are already optimized

### Option 2: Online Conversion Tools

1. **Convertmodel.com**: https://convertmodel.com/
   - Upload ONNX file
   - Select TFLite as output
   - Download converted model

2. **Netron + TensorFlow**: https://netron.app
   - Visualize your ONNX model
   - Export to TensorFlow format
   - Convert to TFLite

### Option 3: Fix TensorFlow Environment

```bash
# Create fresh environment
conda create -n tflite_convert python=3.10
conda activate tflite_convert

# Install dependencies
pip install tensorflow==2.15.0
pip install onnx==1.15.0
pip install onnx-tf==1.10.0

# Run conversion
python src/conversion/tflite_converter.py
```

### Option 4: Use Docker

```dockerfile
FROM tensorflow/tensorflow:2.15.0

RUN pip install onnx onnx-tf

COPY Base-dir/onnx_models /onnx_models
WORKDIR /workspace

# Your conversion script here
```

## Model Files

All models include:
- `{model_name}.onnx` - ONNX format model
- `metadata.json` - Model configuration
- `labels.txt` - Class labels
- `android_metadata.json` - Android-specific config

## Next Steps

1. **For Android Development**: Use ONNX Runtime (Option 1)
2. **If TFLite Required**: Use online tools (Option 2)
3. **For Batch Conversion**: Fix environment (Option 3) or use Docker (Option 4)

## Repository

All conversion scripts are in: `src/conversion/`
- `tflite_converter.py` - Main converter (needs TF fix)
- `simple_onnx_tflite.py` - Simplified version
- Instructions in this file

Push to GitHub for backup and collaboration.
