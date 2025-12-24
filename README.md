# Intelli-PEST Backend

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-red)]()
[![TensorFlow 2.14+](https://img.shields.io/badge/TensorFlow-2.14%2B-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Training Status](https://img.shields.io/badge/Training-Complete-success)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-96.25%25-brightgreen)]()

Complete ML pipeline for **Sugarcane Pest Detection**: Training → Ensemble Models → Knowledge Distillation → Mobile Deployment

## 🎯 Final Results

| Metric | Value |
|--------|-------|
| **Final Validation Accuracy** | **96.25%** |
| Model Size (PyTorch) | 46.85 MB |
| Model Size (ONNX) | 46.50 MB |
| Parameters | 12.2M |
| Training Time | ~22 hours |
| Total Epochs | 250 |

### Per-Class Performance

| Class | Accuracy | Class | Accuracy |
|-------|----------|-------|----------|
| Healthy | 99.28% | army worm | **100.00%** |
| Internode borer | 94.00% | mealy bug | 95.52% |
| Pink borer | 92.05% | porcupine damage | **100.00%** |
| Rat damage | **100.00%** | root borer | 95.65% |
| Stalk borer | 95.24% | termite | 88.89% |
| Top borer | 98.54% | | |

---

## 📊 Project Overview

This repository implements a complete ML pipeline with:
1. **11 Pre-trained Teacher Models** - Base + Ensemble architectures
2. **Sequential Knowledge Distillation** - Train lightweight student using EWC
3. **Multi-Format Export** - PyTorch, ONNX, TFLite for any deployment

### Architecture Flow

```
BASE MODELS (7)          ENSEMBLE MODELS (4)         KNOWLEDGE DISTILLATION
┌─────────────┐          ┌─────────────────┐         ┌──────────────────┐
│ AlexNet     │          │ Ensemble_Attn   │         │                  │
│ MobileNetV2 │          │ Ensemble_Concat │         │  ENHANCED        │
│ EfficientNet│─────────▶│ Ensemble_Cross  │────────▶│  STUDENT         │
│ ResNet50    │          │ Super_Ensemble  │         │  MODEL           │
│ DarkNet53   │          └─────────────────┘         │                  │
│ InceptionV3 │                                      │  96.25% Acc      │
│ YOLO11n-cls │                                      │  46 MB           │
└─────────────┘                                      └──────────────────┘
                    11 ONNX Teachers
```

---

## 📁 Repository Structure

```
Intelli_PEST-Backend/
├── knowledge_distillation/          # 🎓 Sequential distillation pipeline
│   ├── train_sequential.py          # Main training script (recommended)
│   ├── train.py                     # Legacy multi-teacher training
│   ├── finish_training.py           # Final evaluation & export
│   ├── configs/
│   │   └── config.yaml              # Training configuration
│   ├── src/
│   │   ├── enhanced_student_model.py # Student with CBAM, FPN
│   │   ├── teacher_loader.py         # Multi-format loader
│   │   ├── sequential_trainer.py     # EWC-based training
│   │   ├── dataset.py                # Data loading
│   │   ├── evaluator.py              # Metrics & visualization
│   │   └── exporter.py               # Model export
│   └── requirements.txt
│
├── tflite_models_compatible/         # 📱 Pre-trained teacher models
│   ├── onnx_models/                  # 11 ONNX teachers
│   ├── android_models/               # TFLite versions
│   └── model_metadata.json
│
├── src/                              # Core training modules
│   ├── training/                     # Base model training
│   │   ├── base_training.py
│   │   └── ensemble_training.py
│   ├── conversion/                   # PyTorch → ONNX → TFLite
│   └── utils/                        # Utilities
│
├── configs/                          # Global configurations
│   ├── training_config.yaml
│   ├── model_config.yaml
│   └── conversion_config.yaml
│
├── docs/                             # 📚 Documentation
│   ├── COMPLETE_PIPELINE.md          # Full pipeline guide
│   ├── INSTALLATION.md
│   └── TRAINING_GUIDE.md
│
├── scripts/                          # Utility scripts
├── tests/                            # Test files
├── pipeline.py                       # Full pipeline runner
└── run_conversion.py                 # Model conversion script
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/SERVER-246/Intelli_PEST-Backend.git
cd Intelli_PEST-Backend
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r knowledge_distillation/requirements.txt
```

### 3. Prepare Dataset

Organize your dataset in ImageFolder format:
```
IMAGE DATASET/
├── Healthy/
├── Internode borer/
├── Pink borer/
├── Rat damage/
├── Stalk borer/
├── Top borer/
├── army worm/
├── mealy bug/
├── porcupine damage/
├── root borer/
└── termite/
```

Update the path in `knowledge_distillation/configs/config.yaml`:
```yaml
dataset:
  path: "/your/path/to/IMAGE DATASET"
```

### 4. Run Knowledge Distillation

```bash
cd knowledge_distillation

# Sequential training from all 11 teachers
python train_sequential.py --config configs/config.yaml

# After training completes, run final evaluation
python finish_training.py
```

### 5. Use Trained Models

**PyTorch:**
```python
import torch
checkpoint = torch.load('exported_models/student_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

**ONNX:**
```python
import onnxruntime as ort
session = ort.InferenceSession('exported_models/student_model.onnx')
outputs = session.run(None, {'input': image_array})
```

---

## 📋 Complete Pipeline

### Phase 1: Base Model Training
Train 7 base architectures on your dataset:
```bash
python pipeline.py --stage training --data_path /path/to/dataset
```

### Phase 2: Ensemble Creation
Create 4 ensemble models:
```bash
python pipeline.py --stage ensemble
```

### Phase 3: Model Conversion
Convert all to ONNX format:
```bash
python pipeline.py --stage conversion
```

### Phase 4: Knowledge Distillation
Train student model sequentially from all 11 teachers:
```bash
cd knowledge_distillation
python train_sequential.py --config configs/config.yaml
```

**See [docs/COMPLETE_PIPELINE.md](docs/COMPLETE_PIPELINE.md) for detailed instructions.**

---

## 🎓 Knowledge Distillation Details

### Sequential Training Strategy

The student learns from each teacher sequentially with **Elastic Weight Consolidation (EWC)** to prevent forgetting:

```
Phase 1:  AlexNet          (20 epochs)  →  79.30%
Phase 2:  MobileNetV2      (20 epochs)  →  86.42%
Phase 3:  EfficientNet-B0  (20 epochs)  →  88.23%
Phase 4:  ResNet50         (20 epochs)  →  90.43%
Phase 5:  DarkNet53        (20 epochs)  →  91.33%
Phase 6:  InceptionV3      (20 epochs)  →  92.11%
Phase 7:  YOLO11n-cls      (20 epochs)  →  93.79%
Phase 8:  Ensemble_Attn    (20 epochs)  →  94.83%
Phase 9:  Ensemble_Concat  (20 epochs)  →  93.79%
Phase 10: Ensemble_Cross   (20 epochs)  →  50.19%
Phase 11: Super_Ensemble   (20 epochs)  →  92.24%
Final:    Refinement       (30 epochs)  →  96.25% ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 250 epochs                Final: 96.25%
```

### Enhanced Student Architecture

- **Multi-Scale Convolution** (Inception-style stem)
- **CBAM Attention** (Channel + Spatial attention)
- **Inverted Residual Blocks** (MobileNet-style efficiency)
- **Feature Pyramid Network** (Multi-scale features)
- **Knowledge Consolidation Blocks** (Teacher knowledge integration)

### Training Configuration

```yaml
training:
  epochs_per_teacher: 20      # Epochs per teacher phase
  final_ensemble_epochs: 30   # Final refinement
  learning_rate: 0.001
  optimizer: adamw
  scheduler: cosine
  ewc_lambda: 1000           # EWC regularization
  
distillation:
  temperature: 4.0           # Soft label temperature
  alpha: 0.7                 # Soft label weight
  beta: 0.3                  # Hard label weight
```

---

## 📱 Teacher Models (11 Total)

| Model | Type | ONNX Size | Description |
|-------|------|-----------|-------------|
| AlexNet | Base | 222 MB | Classic CNN |
| MobileNetV2 | Base | 3.5 MB | Mobile-efficient |
| EfficientNet-B0 | Base | 5.3 MB | Compound scaling |
| ResNet50 | Base | 24.5 MB | Residual learning |
| DarkNet53 | Base | 20.2 MB | YOLO backbone |
| InceptionV3 | Base | 23.4 MB | Multi-scale |
| YOLO11n-cls | Base | 5.3 MB | Ultralytics |
| Ensemble_Attention | Ensemble | 99.4 MB | Attention fusion |
| Ensemble_Concat | Ensemble | 99.8 MB | Concatenation |
| Ensemble_Cross | Ensemble | 106.9 MB | Cross-attention |
| Super_Ensemble | Ensemble | 144.5 MB | Meta-ensemble |

---

## 📈 Class Labels

```python
CLASS_NAMES = [
    "Healthy",           # 0
    "Internode borer",   # 1
    "Pink borer",        # 2
    "Rat damage",        # 3
    "Stalk borer",       # 4
    "Top borer",         # 5
    "army worm",         # 6
    "mealy bug",         # 7
    "porcupine damage",  # 8
    "root borer",        # 9
    "termite"            # 10
]
```

---

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8 GB VRAM | 16+ GB VRAM |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB | 50 GB |

**Tested on:** NVIDIA RTX 4500 Ada (25.8 GB VRAM)

---

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

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 👥 Authors

- **Intelli-PEST Team** - [SERVER-246](https://github.com/SERVER-246)

---

## 📚 Documentation

- [Complete Pipeline Guide](docs/COMPLETE_PIPELINE.md) - Full step-by-step instructions
- [Installation Guide](docs/INSTALLATION.md) - Setup instructions
- [Training Guide](docs/TRAINING_GUIDE.md) - Training configurations
- [Knowledge Distillation README](knowledge_distillation/README.md) - Detailed KD documentation

---

**⭐ Star this repo if you find it useful!**

---

*Last Updated: December 2024 | Pipeline Version: 2.0.0*
