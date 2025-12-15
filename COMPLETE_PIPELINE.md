# Complete Pest Detection Pipeline: Training to TFLite

This document outlines the **complete reproducible pipeline** for training pest detection models and converting them to TensorFlow Lite format for mobile deployment.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE PIPELINE FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. MODEL TRAINING (src/training/)                             │
│     ├── base_training.py: Train individual models              │
│     │   • MobileNetV2, ResNet50, InceptionV3, EfficientNetB0   │
│     │   • YOLOv11n-cls, DarkNet53, AlexNet                     │
│     │   • Outputs: .pt (JIT) files in Base-dir/deployment_models/
│     │                                                          │
│     └── ensemble_training.py: Create ensemble models           │
│         • Attention-based, Concatenation, Cross-attention     │
│         • Super ensemble combining all methods                 │
│         • Outputs: 4 ensemble models in Base-dir/deployment_models/
│                                                                 │
│  2. ONNX CONVERSION (external/manual or via conversion config)  │
│     • Convert all 11 models: PyTorch → ONNX                    │
│     • Storage: Base-dir/onnx_models/ (fallback for TFLite)     │
│     • Handles adaptive pooling compatibility                   │
│                                                                 │
│  3. TFLITE CONVERSION (src/conversion/)                        │
│     ├── pytorch_to_tflite_quantized.py: Core conversion engine │
│     │   • ONNX → TensorFlow SavedModel → TFLite                │
│     │   • Dynamic Range Quantization (76.6% compression)      │
│     │   • Memory-safe subprocess isolation per model          │
│     │                                                          │
│     └── run_conversion.py: Master reproduction script          │
│         • Single entry point for all 11 models                │
│         • CLI support for batch/single conversion             │
│         • Outputs: tflite_models/ directory                   │
│                                                                 │
│  4. QUALITY ASSURANCE (tests/)                                 │
│     ├── test_training.py: Validate training pipeline           │
│     ├── test_conversion.py: Verify ONNX/TFLite conversion      │
│     └── test_inference.py: Test model inference               │
│                                                                 │
│  FINAL OUTPUT: 11 optimized .tflite models ready for:          │
│  - Android mobile deployment                                   │
│  - Web browser inference (TFJS)                                │
│  - Edge device deployment                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Complete Reproduction Steps

### Phase 1: Setup Environment

```bash
# Clone repository
git clone https://github.com/SERVER-246/Intelli_PEST-Backend
cd Intelli_PEST-Backend

# Create virtual environment
python -m venv venv_tflite

# Activate (Windows)
.\venv_tflite\Scripts\activate

# Activate (Linux/Mac)
source venv_tflite/bin/activate

# Install all dependencies
pip install -r requirements_tflite.txt
```

### Phase 2: Train Base Models

```bash
# Train all base models (requires GPU or will be very slow on CPU)
python -m src.training.base_training \
    --data_path "path/to/training/data" \
    --output_dir "./checkpoints" \
    --epochs 100 \
    --batch_size 32

# Output: 7 base models (.pt files)
```

### Phase 3: Create Ensemble Models

```bash
# Combine base models into ensemble models
python -m src.training.ensemble_training \
    --checkpoint_dir "./checkpoints" \
    --output_dir "./checkpoints" \
    --ensemble_methods "attention,concat,cross,super"

# Output: 4 ensemble models (.pt files)
```

### Phase 4: Export to ONNX (Optional - Pre-converted available)

```bash
# ONNX files are pre-converted and available in Base-dir/onnx_models/
# If you want to create your own ONNX models from PyTorch:

python -c "
import torch
from src.training.base_training import MODELS_CONFIG

for model_name in MODELS_CONFIG:
    model = torch.jit.load(f'checkpoints/{model_name}.pt')
    example_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, example_input, 
        f'onnx_models/{model_name}.onnx',
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )
"
```

### Phase 5: Convert to TFLite (Primary Focus)

```bash
# Convert all 11 models to TFLite with quantization
python run_conversion.py

# Or convert single model
python run_conversion.py --model mobilenet_v2

# Output: 11 .tflite files in tflite_models/ directory
# File sizes: Total 693.01 MB (76.6% compression)
```

### Phase 6: Verify Models

```bash
# Run test suite to verify conversions
python -m pytest tests/test_conversion.py -v
python -m pytest tests/test_inference.py -v

# Check model sizes and metadata
python scripts/check_models.py

# Output: Model verification report
```

## Directory Structure

```
Intelli_PEST-Backend/
├── src/
│   ├── training/
│   │   ├── base_training.py       # Train 7 base models
│   │   ├── ensemble_training.py   # Create 4 ensemble models
│   │   └── __init__.py
│   ├── conversion/
│   │   ├── pytorch_to_tflite_quantized.py  # Core conversion engine
│   │   └── __init__.py
│   └── utils/                     # Utility functions
│
├── configs/
│   ├── training_config.yaml       # Training hyperparameters
│   ├── model_config.yaml          # Model architecture configs
│   └── conversion_config.yaml     # Conversion settings
│
├── scripts/
│   └── check_models.py            # Model verification utility
│
├── tests/
│   ├── test_training.py           # Training validation
│   ├── test_conversion.py         # ONNX/TFLite conversion tests
│   └── test_inference.py          # Inference tests
│
├── docs/
│   ├── INSTALLATION.md            # Setup instructions
│   └── TRAINING_GUIDE.md          # Training details
│
├── tflite_models/                 # Final output (11 models)
│   ├── mobilenet_v2/
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
│
├── checkpoints/                   # Model artifacts from training
├── requirements_tflite.txt        # All 60 dependencies (frozen)
├── run_conversion.py              # Master conversion script
├── setup.py                       # Package configuration
└── README.md                      # Main documentation
```

## Models Included

### Base Models (7)
| Model | Input Size | Params | Performance |
|-------|-----------|--------|-------------|
| MobileNetV2 | 224×224 | 3.5M | Fast, lightweight |
| ResNet50 | 224×224 | 23.5M | Balanced accuracy |
| InceptionV3 | 299×299 | 27.2M | High accuracy |
| EfficientNetB0 | 224×224 | 5.3M | Very efficient |
| YOLOv11n-cls | 224×224 | 2.7M | Real-time capable |
| DarkNet53 | 224×224 | 41.6M | Robust features |
| AlexNet | 224×224 | 60.9M | Classical baseline |

### Ensemble Models (4)
| Method | Description | Size |
|--------|-------------|------|
| Attention | Weighted averaging by attention scores | 99.59 MB |
| Concatenation | Feature concatenation + FC layer | 100.11 MB |
| Cross-Attention | Cross-modal attention fusion | 107.05 MB |
| Super Ensemble | Combines all 3 methods | 145.02 MB |

## TFLite Model Outputs

All models converted with **Dynamic Range Quantization**:

```
Original size:  2.96 GB (all PyTorch models)
TFLite size:    693.01 MB (all quantized models)
Compression:    76.6% average reduction
Inference:      5-100ms per image (device dependent)
```

## Requirements

### System Requirements
- **Python**: 3.10+
- **GPU**: NVIDIA CUDA 12.0+ (recommended for training)
- **Storage**: 10 GB free (models + data)
- **RAM**: 16+ GB (for training), 4+ GB (for conversion)

### Software Dependencies
See `requirements_tflite.txt` for exact versions:
- **PyTorch**: 2.3.1 (model training/loading)
- **TensorFlow**: 2.20.0 (conversion target)
- **ONNX**: 1.16.0 (intermediate format)
- **onnx2tf**: 1.25.15 (ONNX → TF conversion)
- **NumPy**: 1.26.4 (data handling)
- **Plus 55 additional dependencies** (see requirements_tflite.txt)

## Key Features

✅ **Complete Reproducibility**: All steps from training to deployment  
✅ **Memory Efficient**: Subprocess isolation prevents OOM errors  
✅ **Network Tolerant**: Handles adaptive pooling via ONNX fallback  
✅ **Production Ready**: Dynamic Range Quantization for mobile  
✅ **Well Documented**: INSTALLATION.md, TRAINING_GUIDE.md  
✅ **Tested Pipeline**: test_training.py, test_conversion.py, test_inference.py  
✅ **Mobile Friendly**: All 11 models optimized for Android/Web  

## Troubleshooting

### Out of Memory During Training
- Reduce batch size in `configs/training_config.yaml`
- Train models sequentially instead of in parallel

### Conversion Failures
- Check Base-dir/onnx_models/ for pre-converted ONNX files
- These are automatically used as fallback for problematic models

### TFLite Inference Slow
- Verify quantized model is being used (tflite_models/)
- Use GPU delegate or NPU support on target device

## Next Steps

1. **Clone and setup** environment (Phase 1)
2. **Prepare training data** (requires pest detection dataset)
3. **Run training pipeline** (Phases 2-3, optional if using pre-trained)
4. **Run TFLite conversion** (Phase 5, uses pre-trained models)
5. **Verify models** (Phase 6)
6. **Deploy to mobile** (use converted TFLite files)

## Support

For issues or questions:
- Check INSTALLATION.md for setup issues
- See TRAINING_GUIDE.md for training-specific help
- Review README.md for general information

---

**Version**: 1.0  
**Last Updated**: December 15, 2025  
**Maintainer**: SERVER-246
