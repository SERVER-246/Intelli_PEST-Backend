# Intelli_PEST-Backend: PyTorch to TFLite Conversion Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![PyTorch 2.3.1](https://img.shields.io/badge/PyTorch-2.3.1-red)]()
[![TensorFlow 2.20](https://img.shields.io/badge/TensorFlow-2.20-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Production-ready pipeline for converting pest detection models from PyTorch to TensorFlow Lite with Dynamic Range Quantization.**

## ✅ Status: Complete

**All 11 models successfully converted to optimized TFLite format**

| Model | PyTorch Size | TFLite Size | Compression |
|-------|--------------|-------------|-------------|
| mobilenet_v2 | 12.17 MB | 3.18 MB | 73.9% |
| darknet53 | 81.28 MB | 20.45 MB | 74.8% |
| resnet50 | 98.26 MB | 24.83 MB | 74.7% |
| inception_v3 | 104.63 MB | 23.10 MB | 77.9% |
| efficientnet_b0 | 19.19 MB | 5.10 MB | 73.4% |
| yolo11n-cls | 19.18 MB | 5.10 MB | 73.4% |
| alexnet | 171.74 MB | 164.48 MB | 4.2% |
| ensemble_attention | 577.58 MB | 99.59 MB | 82.8% |
| ensemble_concat | 579.58 MB | 100.11 MB | 82.8% |
| ensemble_cross | 621.65 MB | 107.05 MB | 82.8% |
| super_ensemble | 770.28 MB | 145.02 MB | 81.2% |

**Total: 2.96 GB → 693.01 MB (76.6% compression)**

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
python -m venv venv_tflite
```

**Activate (Windows):**
```bash
.\venv_tflite\Scripts\activate
```

**Activate (Linux/macOS):**
```bash
source venv_tflite/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements_tflite.txt
```

### 3. Run Conversion Pipeline

Convert all 11 models:
```bash
python run_conversion.py
```

Convert single model:
```bash
python run_conversion.py --model mobilenet_v2
```

## 📋 Project Structure

```
Intelli_PEST-Backend/
├── run_conversion.py                     # Master script (entry point)
├── requirements_tflite.txt               # Complete dependencies
├── requirements_original.txt             # Original training requirements
├── src/
│   └── conversion/
│       ├── pytorch_to_tflite_quantized.py    # Main conversion logic
│       └── __init__.py
├── tflite_models/                        # Output directory
│   ├── mobilenet_v2/
│   │   └── mobilenet_v2.tflite
│   ├── darknet53/
│   ├── resnet50/
│   ├── inception_v3/
│   ├── efficientnet_b0/
│   ├── yolo11n-cls/
│   ├── alexnet/
│   ├── ensemble_attention/
│   ├── ensemble_concat/
│   ├── ensemble_cross/
│   └── super_ensemble/
├── configs/
├── data/
├── docs/
├── scripts/
├── tests/
└── setup.py
```

## 🔧 Conversion Pipeline Details

### Process Flow

```
PyTorch Model (.pt)
        ↓
   Step 1: Load Model
        ↓
   Step 2: Export to ONNX (opset 11-17)
        ↓
   Step 3: Convert to TensorFlow SavedModel
        ↓
   Step 4: Convert to TFLite (Dynamic Range Quantization)
        ↓
   Step 5: Verify Output
        ↓
TFLite Model (.tflite) - Ready for Mobile Deployment
```

### Quantization Strategy

- **Type**: Dynamic Range Quantization
- **Benefits**: 
  - ~76.6% average model size reduction
  - Minimal accuracy loss
  - CPU-optimized inference
  - No calibration dataset required
- **Output**: Full precision weights, quantized activations

## 💻 Advanced Usage

### Custom Input/Output Directories

```bash
python run_conversion.py \
    --input_dir D:\deployment_models \
    --output_dir ./tflite_output
```

### Verbose Output

```bash
python run_conversion.py --verbose
```

### View Conversion Results

```bash
cat tflite_models/quantized_conversion_report.json
```

## 📦 Output Files

Each model directory contains:
- `{model_name}.tflite` - The optimized TFLite model
- `conversion_result.json` - Conversion metadata
- `android_metadata.json` - Android integration info

## 🔐 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.10+ |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB | 10 GB |
| OS | Windows 10 | Windows 10+, macOS 10.14+, Ubuntu 18.04+ |

## 📚 Dependencies

### Core Frameworks
- **PyTorch**: 2.3.1 (for model loading)
- **TensorFlow**: 2.20.0 (for conversion)
- **ONNX**: 1.16.0 (intermediate format)

### Conversion Tools
- **onnx2tf**: 1.25.15 (ONNX → TensorFlow)
- **onnx-graphsurgeon**: For graph optimization
- **onnxsim**: ONNX simplification

### Scientific Computing
- **NumPy**: 1.26.4 (array operations)
- **SciPy**: For numerical computation

See `requirements_tflite.txt` for complete dependency list with exact versions.

## 🎓 How It Works

1. **Model Loading**: Loads PyTorch JIT-compiled models
2. **ONNX Export**: Exports to ONNX format with fallback to pre-converted files
3. **TensorFlow Conversion**: Converts ONNX to TensorFlow SavedModel
4. **TFLite Conversion**: Applies Dynamic Range Quantization
5. **Verification**: Validates output shape and inference capability
6. **Reporting**: Generates detailed conversion statistics

## ⚠️ Known Issues & Solutions

### Issue 1: Memory Errors for Large Models
**Cause**: Insufficient RAM during conversion
**Solution**: Increase RAM or convert on cloud instance

### Issue 2: AdaptiveAvgPool2d ONNX Export
**Cause**: Some PyTorch models use adaptive pooling incompatible with ONNX
**Solution**: Script uses pre-converted ONNX files as fallback

### Issue 3: Network Timeouts
**Cause**: onnx2tf tries to download test data
**Solution**: Script patches this function to use dummy data

## 📊 Performance Metrics

### Conversion Time
- Small models (< 50MB): ~30-60 seconds
- Large models (> 500MB): 2-5 minutes
- Total for 11 models: ~7-10 minutes

### Inference Performance
- TFLite: 50-100ms per inference (CPU, mobile)
- TFLite: 10-20ms per inference (with GPU delegate)

## 🔗 Integration

### Android Integration
```kotlin
val interpreter = Interpreter(tfliteModelFile)
val inputArray = FloatArray(256 * 256 * 3)
val outputArray = FloatArray(11) // 11 pest classes
interpreter.run(arrayOf(inputArray), arrayOf(outputArray))
```

### Web Deployment
```javascript
import * as tf from '@tensorflow/tfjs';
import * as tflite from '@tensorflow/tfjs-tflite';

const model = await tflite.loadTFLiteModel('file://model.tflite');
```

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{intellipest_backend,
  author = {Your Name},
  title = {Intelli_PEST-Backend: PyTorch to TFLite Conversion Pipeline},
  year = {2025},
  url = {https://github.com/SERVER-246/Intelli_PEST-Backend}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

For issues, questions, or suggestions:
- Open an [Issue](https://github.com/SERVER-246/Intelli_PEST-Backend/issues)
- Email: singh.sugam.47@gmail.com

---

**Last Updated**: December 15, 2025  
**Status**: ✅ Production Ready  
**Python**: 3.10+  
**PyTorch**: 2.3.1 | **TensorFlow**: 2.20.0 | **ONNX**: 1.16.0
