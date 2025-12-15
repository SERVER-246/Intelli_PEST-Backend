# Intelli_PEST-Backend: Complete ML Pipeline - Training to TFLite Deployment

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![PyTorch 2.3.1](https://img.shields.io/badge/PyTorch-2.3.1-red)]()
[![TensorFlow 2.20](https://img.shields.io/badge/TensorFlow-2.20-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Complete, production-ready ML pipeline for pest detection: from model training → ensemble creation → ONNX export → TensorFlow Lite conversion with Dynamic Range Quantization.**

**This repository contains the entire reproducible pipeline to train pest detection models from scratch and convert them to optimized TFLite format for mobile and edge deployment.**

## ✅ Pipeline Status: Complete & Reproducible

**All pipeline stages fully implemented:**
- ✅ Base model training (7 models)
- ✅ Ensemble model creation (4 models)  
- ✅ ONNX conversion (with fallback mechanism)
- ✅ TFLite conversion (all 11 models)
- ✅ Dynamic Range Quantization
- ✅ Test suite for validation

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

## 📋 Complete Project Structure

```
Intelli_PEST-Backend/
│
├── 📄 COMPLETE_PIPELINE.md               # Full pipeline documentation
├── 📄 run_conversion.py                  # Master TFLite conversion script
├── 📄 requirements_tflite.txt            # All 60 dependencies (frozen)
├── 📄 setup.py                           # Package configuration
│
├── src/
│   ├── training/                         # MODEL TRAINING STAGE
│   │   ├── base_training.py              # Train 7 individual models
│   │   ├── ensemble_training.py          # Create 4 ensemble models
│   │   └── __init__.py
│   │
│   ├── conversion/                       # TFLITE CONVERSION STAGE
│   │   ├── pytorch_to_tflite_quantized.py    # Core conversion engine
│   │   └── __init__.py
│   │
│   ├── deployment/                       # Deployment utilities
│   │   └── __init__.py
│   │
│   └── utils/                            # Shared utilities
│
├── configs/                              # Configuration files
│   ├── training_config.yaml              # Training hyperparameters
│   ├── model_config.yaml                 # Model architectures
│   └── conversion_config.yaml            # Conversion settings
│
├── docs/                                 # Documentation
│   ├── INSTALLATION.md                   # Environment setup
│   └── TRAINING_GUIDE.md                 # Training instructions
│
├── scripts/                              # Utility scripts
│   └── check_models.py                   # Model verification
│
├── tests/                                # Test suite
│   ├── test_training.py                  # Training validation
│   ├── test_conversion.py                # Conversion tests
│   └── test_inference.py                 # Inference tests
│
└── tflite_models/                        # FINAL OUTPUT (Phase 5)
    ├── mobilenet_v2/
    │   ├── mobilenet_v2.tflite           # Optimized model
    │   ├── conversion_result.json        # Metadata
    │   └── android_metadata.json         # Android config
    ├── darknet53/
    ├── resnet50/
    ├── inception_v3/
    ├── efficientnet_b0/
    ├── yolo11n-cls/
    ├── alexnet/
    ├── ensemble_attention/
    ├── ensemble_concat/
    ├── ensemble_cross/
    └── super_ensemble/
```

## 🔄 Complete Pipeline Stages

### **STAGE 1: Model Training** (Optional - Pre-trained models available)
```bash
python -m src.training.base_training \
    --data_path "path/to/data" \
    --output_dir "./checkpoints" \
    --epochs 100
```
**Outputs 7 models**: MobileNetV2, ResNet50, InceptionV3, EfficientNetB0, YOLOv11n-cls, DarkNet53, AlexNet

### **STAGE 2: Ensemble Model Creation** (Optional - Pre-trained models available)
```bash
python -m src.training.ensemble_training \
    --checkpoint_dir "./checkpoints" \
    --output_dir "./checkpoints"
```
**Outputs 4 ensemble models**: Attention, Concatenation, Cross-Attention, Super Ensemble

### **STAGE 3: ONNX Conversion** (Pre-converted models available in Base-dir/onnx_models/)
- Converts PyTorch models to ONNX intermediate format
- Includes fallback mechanism for adaptive pooling compatibility
- Stored in: `Base-dir/onnx_models/` for re-use and verification

### **STAGE 4: TFLite Conversion** (Main focus - FULLY AUTOMATED)
```bash
python run_conversion.py
```
**Converts all 11 models** with Dynamic Range Quantization

### **STAGE 5: Validation & Testing**
```bash
python -m pytest tests/
python scripts/check_models.py
```
**Verifies all conversions** and model integrity

## 🚀 Quick Start Guide

### For Users With Pre-Trained Models (TFLite Conversion Only)

```bash
# Step 1: Clone repository
git clone https://github.com/SERVER-246/Intelli_PEST-Backend
cd Intelli_PEST-Backend

# Step 2: Create environment
python -m venv venv_tflite
.\venv_tflite\Scripts\activate  # Windows
# OR
source venv_tflite/bin/activate  # Linux/Mac

# Step 3: Install dependencies
pip install -r requirements_tflite.txt

# Step 4: Run TFLite conversion
python run_conversion.py

# Step 5: Check outputs
ls tflite_models/  # All 11 .tflite files
```

### For Researchers (Complete Pipeline from Training)

```bash
# Follow installation in docs/INSTALLATION.md
# Run training in docs/TRAINING_GUIDE.md
# Then follow TFLite conversion above
```

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
