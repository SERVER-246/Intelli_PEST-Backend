# Intelli-PEST Backend

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-red)]()
[![TensorFlow 2.14+](https://img.shields.io/badge/TensorFlow-2.14%2B-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Complete ML pipeline for sugarcane pest detection: **Training → Ensemble Models → Knowledge Distillation → Mobile Deployment**

## 🎯 Overview

This repository contains:
1. **11 Pre-trained Teacher Models** - Base models + ensemble architectures
2. **Knowledge Distillation Pipeline** - Train lightweight student models
3. **Multi-Format Export** - PyTorch, ONNX, TFLite for any deployment

## 📊 Model Summary

### Teacher Models (11 Total)

| Model | Type | TFLite Size | Classes |
|-------|------|-------------|---------|
| MobileNetV2 | Base | 3.18 MB | 11 |
| ResNet50 | Base | 24.83 MB | 11 |
| InceptionV3 | Base | 23.10 MB | 11 |
| EfficientNet-B0 | Base | 5.11 MB | 11 |
| DarkNet53 | Base | 20.46 MB | 11 |
| AlexNet | Base | 164.48 MB | 11 |
| YOLO11n-cls | Base | 5.11 MB | 11 |
| Ensemble-Attention | Ensemble | 94.98 MB | 11 |
| Ensemble-Concat | Ensemble | 95.48 MB | 11 |
| Ensemble-Cross | Ensemble | 102.09 MB | 11 |
| Super-Ensemble | Ensemble | 138.30 MB | 11 |

### Student Model (Knowledge Distillation)

| Metric | Value |
|--------|-------|
| Architecture | Custom CNN |
| Parameters | ~1.3M |
| Model Size | ~5 MB |
| Input Size | 256×256×3 |
| Output Classes | 12 |

## 📁 Repository Structure

```
Intelli_PEST-Backend/
├── knowledge_distillation/          # 🎓 Student model training
│   ├── train.py                     # Main training script
│   ├── configs/config.yaml          # Training configuration
│   ├── src/                         # Source modules
│   │   ├── student_model.py         # CNN architecture
│   │   ├── dataset.py               # Data loading
│   │   ├── trainer.py               # KD training loop
│   │   ├── evaluator.py             # Metrics & visualization
│   │   └── exporter.py              # Model export
│   └── models/student/              # Exported models
│       ├── pytorch/                 # .pt files
│       ├── onnx/                    # .onnx files
│       └── tflite/                  # .tflite files
│
├── tflite_models_compatible/        # 📱 Android-ready models
│   ├── android_models/              # TFLite models (11 total)
│   ├── onnx_models/                 # ONNX models (11 total)
│   └── model_metadata.json          # Model information
│
├── src/                             # Core training modules
│   ├── training/                    # Base model training
│   └── ensemble/                    # Ensemble creation
│
├── scripts/                         # Utility scripts
│   ├── verify_models.py             # Model verification
│   └── check_models.py              # Quick model check
│
└── configs/                         # Configuration files
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/SERVER-246/Intelli_PEST-Backend.git
cd Intelli_PEST-Backend
```

### 2. Install Dependencies

```bash
pip install torch torchvision onnx onnxruntime-gpu tensorflow
pip install -r knowledge_distillation/requirements.txt
```

### 3. Train Student Model (Knowledge Distillation)

```bash
cd knowledge_distillation

# Update dataset path in configs/config.yaml
# Then run training:
python train.py --config configs/config.yaml
```

### 4. Use Pre-trained Models

**TFLite (Android):**
```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="tflite_models_compatible/android_models/mobilenet_v2.tflite"
)
interpreter.allocate_tensors()
```

**ONNX (Cross-platform):**
```python
import onnxruntime as ort

session = ort.InferenceSession(
    "tflite_models_compatible/onnx_models/mobilenet_v2.onnx"
)
```

## 🎓 Knowledge Distillation

Train a lightweight model using knowledge from all 11 teachers:

```bash
cd knowledge_distillation
python train.py --epochs 100 --batch_size 32
```

**What it does:**
- Loads 11 teacher models (ONNX format)
- Trains student with soft labels + hard labels
- Exports to PyTorch, ONNX, TFLite
- Generates evaluation metrics & plots

**Output:**
- `models/student/pytorch/student_model.pt`
- `models/student/onnx/student_model.onnx`
- `models/student/tflite/student_model.tflite`
- `plots/confusion_matrix.png`
- `metrics/classification_report.json`

See [knowledge_distillation/README.md](knowledge_distillation/README.md) for details.

## 📱 Class Labels

### Teacher Models (11 Classes)
```
0: Healthy           5: Top borer
1: Internode borer   6: Army worm
2: Pink borer        7: Mealy bug
3: Rat damage        8: Porcupine damage
4: Stalk borer       9: Root borer
                    10: Termite
```

### Student Model (12 Classes)
```
0: Fall army worm    6: Top borer
1: Healthy           7: Army worm
2: Internode borer   8: Mealy bug
3: Pink borer        9: Porcupine damage
4: Rat damage       10: Root borer
5: Stalk borer      11: Termite
```

## 🔧 Pipeline Commands

### Full Training Pipeline

```bash
# 1. Train base models (optional - pre-trained available)
python pipeline.py --stage training --data_path /path/to/data

# 2. Create ensemble models (optional - pre-trained available)
python pipeline.py --stage ensemble

# 3. Convert to TFLite
python pipeline.py --stage conversion

# 4. Validate all models
python pipeline.py --stage validation
```

### Knowledge Distillation Only

```bash
cd knowledge_distillation
python train.py --config configs/config.yaml --epochs 100
```

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
tensorflow>=2.14.0
onnx>=1.14.0
onnxruntime-gpu>=1.15.0
numpy>=1.24.0
Pillow>=9.0.0
PyYAML>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

## 📈 Performance

| Model | Size | Accuracy* | Inference |
|-------|------|-----------|-----------|
| Super-Ensemble | 138 MB | ~95% | ~200ms |
| EfficientNet-B0 | 5 MB | ~88% | ~30ms |
| Student (KD) | ~5 MB | ~85%+ | ~25ms |

*Accuracy varies by dataset and training configuration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 👥 Authors

- **Intelli-PEST Team** - [SERVER-246](https://github.com/SERVER-246)

---

**⭐ Star this repo if you find it useful!**
