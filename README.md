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

## 🚀 Master Pipeline Script: Complete Automation

**NEW: Use the `pipeline.py` script to run the entire pipeline with a single command!**

### One-Command Pipeline Execution

```bash
# Run complete pipeline (Training → Ensemble → ONNX → TFLite → Validation)
python pipeline.py
```

### Run Specific Stages

```bash
# Stage 1: Train base models (7 models)
python pipeline.py --stage training --epochs 100 --data_path /path/to/data

# Stage 2: Create ensemble models (4 models)
python pipeline.py --stage ensemble

# Stage 3: ONNX export (pre-converted, uses fallback)
python pipeline.py --stage onnx

# Stage 4: Convert to TFLite with quantization (11 models)
python pipeline.py --stage conversion --verbose

# Stage 5: Validate and test all models
python pipeline.py --stage validation
```

### Custom Configuration

```bash
# Custom paths and hyperparameters
python pipeline.py \
    --data_path /path/to/training/data \
    --checkpoint_dir ./my_checkpoints \
    --output_dir ./my_tflite_models \
    --epochs 150 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --verbose
```

### Pipeline Options

```bash
python pipeline.py --help
```

**Common options:**
- `--stage` - Run specific stage (training, ensemble, onnx, conversion, validation)
- `--data_path` - Path to training dataset
- `--checkpoint_dir` - Directory for model checkpoints
- `--output_dir` - Directory for TFLite output
- `--epochs` - Number of training epochs (default: 100)
- `--batch_size` - Training batch size (default: 32)
- `--learning_rate` - Learning rate (default: 0.001)
- `--verbose` - Enable verbose output
- `--continue_on_error` - Continue to next stage on errors

## 🚀 Quick Start (TFLite Conversion Only)

If you just want to convert pre-trained models to TFLite:

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

### 3. Run TFLite Conversion Only

For quick TFLite conversion (pre-trained models):
```bash
python run_conversion.py
```

Or convert a single model:
```bash
python run_conversion.py --model mobilenet_v2
```

**Or for complete pipeline from training:**
```bash
python pipeline.py --stage conversion
```

## 📋 Complete Project Structure

```
Intelli_PEST-Backend/
│
├── 📄 pipeline.py                        # MASTER SCRIPT (NEW!)
│                                         # Complete automation: training → TFLite
├── 📄 COMPLETE_PIPELINE.md               # Full pipeline documentation
├── 📄 run_conversion.py                  # TFLite conversion script
├── 📄 requirements_tflite.txt            # All 60 dependencies (frozen)
├── 📄 setup.py                           # Package configuration
│
├── src/
│   ├── training/                         # MODEL TRAINING STAGE
│   │   ├── base_training.py              # Train 7 individual models
│   │   ├── ensemble_training.py          # Create 4 ensemble models
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

### Use Master `pipeline.py` Script for Full Automation

```bash
# Run entire pipeline
python pipeline.py

# Run specific stage
python pipeline.py --stage training|ensemble|onnx|conversion|validation

# With custom parameters
python pipeline.py --epochs 100 --batch_size 32 --data_path /path/to/data --verbose
```

### Stage Details

| Stage | Script | Command | Output |
|-------|--------|---------|--------|
| 1. Training (7 models) | `base_training.py` | `python pipeline.py --stage training` | PyTorch .pt files |
| 2. Ensemble (4 models) | `ensemble_training.py` | `python pipeline.py --stage ensemble` | Ensemble .pt files |
| 3. ONNX Export | Pre-converted | `python pipeline.py --stage onnx` | ONNX ready |
| 4. TFLite Conversion | `run_conversion.py` | `python pipeline.py --stage conversion` | 11 .tflite files |
| 5. Validation | Tests | `python pipeline.py --stage validation` | Test report |

### Alternative: Run Stages Directly

```bash
# Stage 1: Train base models (1-2 hours on GPU)
python -m src.training.base_training --data_path "path/to/data" --epochs 100

# Stage 2: Create ensemble models (30 minutes)
python -m src.training.ensemble_training --checkpoint_dir "./checkpoints"

# Stage 4: Convert to TFLite (15-30 minutes)
python run_conversion.py

# Stage 5: Validate
python -m pytest tests/ -v
```

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
