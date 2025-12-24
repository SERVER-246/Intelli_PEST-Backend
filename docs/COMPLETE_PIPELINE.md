# Complete Intelli-PEST Pipeline Documentation

## From Zero to Deployed Model - A Comprehensive Guide

**Project:** Sugarcane Pest Detection using Deep Learning  
**Pipeline:** Training → Ensemble → Knowledge Distillation → Mobile Deployment  
**Final Achievement:** 96.25% Validation Accuracy with 11-Teacher Sequential Distillation

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Phase 1: Base Model Training](#3-phase-1-base-model-training)
4. [Phase 2: Ensemble Model Creation](#4-phase-2-ensemble-model-creation)
5. [Phase 3: Model Conversion (PyTorch → ONNX → TFLite)](#5-phase-3-model-conversion)
6. [Phase 4: Knowledge Distillation](#6-phase-4-knowledge-distillation)
7. [Final Results](#7-final-results)
8. [Reproducing the Pipeline](#8-reproducing-the-pipeline)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Project Overview

### 1.1 Objective
Create a lightweight, mobile-deployable model for detecting 11 types of sugarcane pests/conditions using knowledge distillation from multiple expert teacher models.

### 1.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INTELLI-PEST PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHASE 1: BASE MODELS           PHASE 2: ENSEMBLES                     │
│  ┌─────────────────┐            ┌─────────────────┐                    │
│  │ • AlexNet       │            │ • Ensemble      │                    │
│  │ • MobileNetV2   │───────────▶│   Attention     │                    │
│  │ • EfficientNet  │            │ • Ensemble      │                    │
│  │ • ResNet50      │            │   Concat        │                    │
│  │ • DarkNet53     │            │ • Ensemble      │                    │
│  │ • InceptionV3   │            │   Cross         │                    │
│  │ • YOLO11n-cls   │            │ • Super         │                    │
│  └─────────────────┘            │   Ensemble      │                    │
│         │                       └─────────────────┘                    │
│         │                              │                               │
│         └──────────────┬───────────────┘                               │
│                        │                                               │
│                        ▼                                               │
│  PHASE 3: CONVERSION TO ONNX                                           │
│  ┌─────────────────────────────────────────┐                           │
│  │ 11 ONNX Teacher Models                  │                           │
│  │ (Cross-platform, verified)              │                           │
│  └─────────────────────────────────────────┘                           │
│                        │                                               │
│                        ▼                                               │
│  PHASE 4: KNOWLEDGE DISTILLATION                                       │
│  ┌─────────────────────────────────────────┐                           │
│  │ Sequential Training (250 epochs total)  │                           │
│  │ • 20 epochs × 11 teachers = 220 epochs  │                           │
│  │ • 30 epochs final refinement            │                           │
│  │ • EWC to prevent forgetting             │                           │
│  └─────────────────────────────────────────┘                           │
│                        │                                               │
│                        ▼                                               │
│  FINAL OUTPUT                                                          │
│  ┌─────────────────────────────────────────┐                           │
│  │ Enhanced Student Model                  │                           │
│  │ • 96.25% Accuracy                       │                           │
│  │ • 12.2M Parameters                      │                           │
│  │ • ~46 MB Size                           │                           │
│  │ • Exports: .pt, .onnx, .tflite          │                           │
│  └─────────────────────────────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Final Model Specifications

| Specification | Value |
|---------------|-------|
| Architecture | Enhanced CNN with CBAM, FPN, Knowledge Consolidation |
| Parameters | 12,200,000 (~12.2M) |
| Model Size (PyTorch) | 46.85 MB |
| Model Size (ONNX) | 46.50 MB |
| Input Size | 256 × 256 × 3 |
| Output Classes | 11 |
| Final Accuracy | **96.25%** |

---

## 2. Dataset Preparation

### 2.1 Dataset Structure

```
IMAGE DATASET/
├── Healthy/              # 139 images
├── Internode borer/      # 350 images
├── Pink borer/           # 352 images
├── Rat damage/           # 312 images
├── Stalk borer/          # 336 images
├── Top borer/            # 548 images
├── army worm/            # 336 images
├── mealy bug/            # 536 images
├── porcupine damage/     # 340 images
├── root borer/           # 368 images
└── termite/              # 324 images
```

### 2.2 Dataset Statistics

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 3,067 | 80% |
| Validation | 773 | 20% |
| **Total** | **3,840** | 100% |

### 2.3 Class Distribution (11 Classes)

| Index | Class Name | Training | Validation |
|-------|------------|----------|------------|
| 0 | Healthy | 111 | 28 |
| 1 | Internode borer | 280 | 70 |
| 2 | Pink borer | 282 | 70 |
| 3 | Rat damage | 250 | 62 |
| 4 | Stalk borer | 269 | 67 |
| 5 | Top borer | 438 | 110 |
| 6 | army worm | 269 | 67 |
| 7 | mealy bug | 429 | 107 |
| 8 | porcupine damage | 272 | 68 |
| 9 | root borer | 294 | 74 |
| 10 | termite | 173 | 50 |

---

## 3. Phase 1: Base Model Training

### 3.1 Models Trained

Seven base architectures were trained using transfer learning:

| Model | Pretrained On | Fine-tuned Accuracy |
|-------|---------------|---------------------|
| AlexNet | ImageNet | ~75% |
| MobileNetV2 | ImageNet | ~85% |
| EfficientNet-B0 | ImageNet | ~88% |
| ResNet50 | ImageNet | ~87% |
| DarkNet53 | ImageNet | ~86% |
| InceptionV3 | ImageNet | ~86% |
| YOLO11n-cls | Custom | ~84% |

### 3.2 Training Configuration

```yaml
training:
  epochs_head: 40          # Classifier head training
  epochs_finetune: 25      # Full model fine-tuning
  batch_size: 32
  learning_rate: 0.001
  optimizer: AdamW
  scheduler: CosineAnnealing
  augmentation:
    - RandomResizedCrop(256)
    - RandomHorizontalFlip
    - ColorJitter
    - RandomRotation(15)
```

### 3.3 Running Base Model Training

```bash
# Train a single base model
python src/training/base_training.py --model resnet50 --epochs 65

# Train all base models
python pipeline.py --stage training --data_path /path/to/dataset
```

---

## 4. Phase 2: Ensemble Model Creation

### 4.1 Ensemble Architectures

Four ensemble models were created combining the base models:

| Ensemble | Fusion Method | Components | Accuracy |
|----------|---------------|------------|----------|
| Ensemble_Attention | Attention Weights | All 7 base models | ~92% |
| Ensemble_Concat | Feature Concatenation | All 7 base models | ~91% |
| Ensemble_Cross | Cross-Attention | All 7 base models | ~90% |
| Super_Ensemble | Meta-Ensemble | All 3 ensembles | ~95% |

### 4.2 Fusion Methods Explained

**Attention Fusion:**
```python
# Learns importance weights for each model
attention_weights = softmax(linear(concat(features)))
fused = sum(attention_weights * features)
```

**Concatenation Fusion:**
```python
# Simple feature concatenation with dimensionality reduction
fused = linear(concat(feature1, feature2, ..., featureN))
```

**Cross-Attention Fusion:**
```python
# Models attend to each other's features
Q, K, V = linear(features)
attention = softmax(Q @ K.T / sqrt(d)) @ V
```

### 4.3 Running Ensemble Training

```bash
# Create ensemble models
python pipeline.py --stage ensemble

# Or individually
python src/training/ensemble_training.py --type attention
python src/training/ensemble_training.py --type concat
python src/training/ensemble_training.py --type cross
```

---

## 5. Phase 3: Model Conversion

### 5.1 Conversion Pipeline

```
PyTorch (.pt/.pth) → ONNX (.onnx) → TFLite (.tflite)
```

### 5.2 ONNX Conversion

All 11 models were converted to ONNX format:

```python
import torch
import torch.onnx

# Load PyTorch model
model = load_model('model.pt')
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 256, 256)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=13
)
```

### 5.3 Converted Models (11 Total)

| Model | ONNX Size | TFLite Size |
|-------|-----------|-------------|
| alexnet.onnx | 222.24 MB | 164.48 MB |
| mobilenet_v2.onnx | 3.47 MB | 3.18 MB |
| efficientnet_b0.onnx | 5.31 MB | 5.11 MB |
| resnet50.onnx | 24.51 MB | 24.83 MB |
| darknet53.onnx | 20.24 MB | 20.46 MB |
| inception_v3.onnx | 23.35 MB | 23.10 MB |
| yolo11n-cls.onnx | 5.33 MB | 5.11 MB |
| ensemble_attention.onnx | 99.35 MB | 94.98 MB |
| ensemble_concat.onnx | 99.81 MB | 95.48 MB |
| ensemble_cross.onnx | 106.87 MB | 102.09 MB |
| super_ensemble.onnx | 144.54 MB | 138.30 MB |

### 5.4 Running Conversion

```bash
# Convert all models
python pipeline.py --stage conversion

# Or use the dedicated script
python run_conversion.py --models all --output tflite_models_compatible/
```

---

## 6. Phase 4: Knowledge Distillation

### 6.1 Why Knowledge Distillation?

- **Problem:** Large ensemble models (138+ MB) are impractical for mobile deployment
- **Solution:** Train a smaller student model to mimic the combined knowledge of all teachers
- **Result:** Compact model (~46 MB) with competitive accuracy (96.25%)

### 6.2 Enhanced Student Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  ENHANCED STUDENT MODEL                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ STEM: Multi-Scale Convolution (Inception-style)         │   │
│  │ • 1×1 conv branch                                       │   │
│  │ • 3×3 conv branch                                       │   │
│  │ • 5×5 conv branch (as 2×3×3)                            │   │
│  │ • Max pool + 1×1 conv branch                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ INVERTED RESIDUAL BLOCKS (MobileNet-style)              │   │
│  │ • Expansion (1×1) → Depthwise (3×3) → Projection (1×1)  │   │
│  │ • 4 stages with increasing channels                     │   │
│  │ • Residual connections                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ CBAM ATTENTION (Channel + Spatial)                      │   │
│  │ • Channel: Global pooling → MLP → Sigmoid               │   │
│  │ • Spatial: Conv → Sigmoid                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ FEATURE PYRAMID NETWORK (FPN)                           │   │
│  │ • Top-down pathway with lateral connections             │   │
│  │ • Multi-scale feature fusion                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ KNOWLEDGE CONSOLIDATION BLOCKS                          │   │
│  │ • Self-attention for integrating teacher knowledge      │   │
│  │ • Feed-forward network                                  │   │
│  │ • Layer normalization                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ CLASSIFIER HEAD                                         │   │
│  │ • Global Average Pooling                                │   │
│  │ • Dropout (0.3)                                         │   │
│  │ • Linear (channels → 11 classes)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Sequential Training Strategy

Unlike traditional multi-teacher distillation (all teachers simultaneously), we use **sequential learning**:

```
Phase 1/11:  AlexNet         (20 epochs)  →  79.30% accuracy
Phase 2/11:  MobileNetV2     (20 epochs)  →  86.42% accuracy
Phase 3/11:  EfficientNet-B0 (20 epochs)  →  88.23% accuracy
Phase 4/11:  ResNet50        (20 epochs)  →  90.43% accuracy
Phase 5/11:  DarkNet53       (20 epochs)  →  91.33% accuracy
Phase 6/11:  InceptionV3     (20 epochs)  →  92.11% accuracy
Phase 7/11:  YOLO11n-cls     (20 epochs)  →  93.79% accuracy
Phase 8/11:  Ensemble_Attn   (20 epochs)  →  94.83% accuracy
Phase 9/11:  Ensemble_Concat (20 epochs)  →  93.79% accuracy*
Phase 10/11: Ensemble_Cross  (20 epochs)  →  50.19%** (noisy)
Phase 11/11: Super_Ensemble  (20 epochs)  →  92.24% accuracy
───────────────────────────────────────────────────────────────
Final Phase: Refinement      (30 epochs)  →  96.25% accuracy
═══════════════════════════════════════════════════════════════
Total: 250 epochs            Final: 96.25%
```

*Note: Accuracy temporarily dips due to some teachers having conflicting knowledge, but EWC prevents catastrophic forgetting.

### 6.4 Elastic Weight Consolidation (EWC)

EWC prevents the student from forgetting what it learned from previous teachers:

```python
# After training with teacher i
fisher_matrix = compute_fisher_information(model, data)
optimal_params = copy(model.parameters())

# During training with teacher i+1
ewc_loss = sum(fisher * (param - optimal_param)^2)
total_loss = distillation_loss + lambda * ewc_loss
```

### 6.5 Loss Function

```python
# Combined distillation loss
L_total = α × L_soft + β × L_hard + λ × L_ewc

Where:
- L_soft  = KL_divergence(student_soft, teacher_soft)   # Soft labels (T=4.0)
- L_hard  = CrossEntropy(student, ground_truth)         # Hard labels
- L_ewc   = EWC_penalty                                 # Prevent forgetting
- α = 0.7, β = 0.3, λ = 1000
```

### 6.6 Training Configuration

```yaml
# configs/config.yaml
dataset:
  path: "G:/AI work/IMAGE DATASET"
  image_size: 256
  batch_size: 16
  train_split: 0.8
  num_workers: 4

student:
  num_classes: 11
  size: "medium"        # small, medium, large
  base_channels: 48
  expand_ratio: 4
  dropout_rate: 0.3
  use_fpn: true

teachers:
  models_dir: "tflite_models_compatible/onnx_models"
  teacher_num_classes: 11
  models:
    - name: "alexnet"
      file: "alexnet.onnx"
      weight: 0.8
    - name: "mobilenet_v2"
      file: "mobilenet_v2.onnx"
      weight: 1.0
    # ... (all 11 teachers)
    - name: "super_ensemble"
      file: "super_ensemble.onnx"
      weight: 2.0

training:
  epochs_per_teacher: 20
  final_ensemble_epochs: 30
  learning_rate: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
  ewc_lambda: 1000
  
distillation:
  temperature: 4.0
  alpha: 0.7    # Soft label weight
  beta: 0.3     # Hard label weight
```

### 6.7 Running Knowledge Distillation

```bash
cd knowledge_distillation

# Run sequential training (recommended)
python train_sequential.py --config configs/config.yaml

# With custom output directory
python train_sequential.py --config configs/config.yaml --output-dir ./output

# Resume from checkpoint
python train_sequential.py --resume checkpoints/checkpoint_after_resnet50.pth

# Final evaluation and export (after training)
python finish_training.py
```

---

## 7. Final Results

### 7.1 Overall Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **96.25%** |
| Total Training Time | ~22 hours |
| Total Epochs | 250 |
| GPU Used | NVIDIA RTX 4500 Ada (25.8 GB) |

### 7.2 Per-Class Accuracy

| Class | Accuracy | Samples |
|-------|----------|---------|
| Healthy | 99.28% | 139 |
| Internode borer | 94.00% | 50 |
| Pink borer | 92.05% | 88 |
| **Rat damage** | **100.00%** | 62 |
| Stalk borer | 95.24% | 63 |
| Top borer | 98.54% | 137 |
| **army worm** | **100.00%** | 67 |
| mealy bug | 95.52% | 67 |
| **porcupine damage** | **100.00%** | 68 |
| root borer | 95.65% | 46 |
| termite | 88.89% | 36 |

### 7.3 Per-Teacher Learning Progress

| Teacher | Best Accuracy During Phase | Final Val Acc |
|---------|---------------------------|---------------|
| AlexNet | 79.30% | 79.30% |
| MobileNetV2 | 86.42% | 83.31% |
| EfficientNet-B0 | 88.23% | 84.22% |
| ResNet50 | 90.43% | 86.68% |
| DarkNet53 | 91.33% | 90.43% |
| InceptionV3 | 92.11% | 90.17% |
| YOLO11n-cls | 93.79% | 91.98% |
| Ensemble_Attention | 94.83% | 92.37% |
| Ensemble_Concat | 93.79% | 92.76% |
| Ensemble_Cross | 50.19% | 45.41% |
| Super_Ensemble | 92.24% | 86.03% |
| **Final Refinement** | **96.25%** | **96.25%** |

### 7.4 Exported Models

| Format | File | Size |
|--------|------|------|
| PyTorch | student_model.pt | 46.85 MB |
| ONNX | student_model.onnx | 46.50 MB |
| TFLite | student_model.tflite | ~46 MB* |

*TFLite conversion pending due to Keras version compatibility

### 7.5 Model Comparison

| Model | Size | Accuracy | Inference Time |
|-------|------|----------|----------------|
| Super_Ensemble (Teacher) | 138 MB | ~95% | ~200ms |
| EfficientNet-B0 (Single) | 5 MB | ~88% | ~30ms |
| **Student (Distilled)** | **46 MB** | **96.25%** | **~40ms** |

---

## 8. Reproducing the Pipeline

### 8.1 Prerequisites

```bash
# Hardware Requirements
- GPU: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- RAM: 16GB minimum
- Storage: 20GB for models and data

# Software Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
```

### 8.2 Installation

```bash
# Clone repository
git clone https://github.com/SERVER-246/Intelli_PEST-Backend.git
cd Intelli_PEST-Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r knowledge_distillation/requirements.txt
```

### 8.3 Prepare Dataset

```bash
# Dataset should be in ImageFolder format
# Update path in configs/config.yaml:
dataset:
  path: "/path/to/your/IMAGE DATASET"
```

### 8.4 Download Teacher Models

Teacher ONNX models should be placed in:
```
tflite_models_compatible/onnx_models/
├── alexnet.onnx
├── mobilenet_v2.onnx
├── efficientnet_b0.onnx
├── resnet50.onnx
├── darknet53.onnx
├── inception_v3.onnx
├── yolo11n-cls.onnx
├── ensemble_attention.onnx
├── ensemble_concat.onnx
├── ensemble_cross.onnx
└── super_ensemble.onnx
```

### 8.5 Run Complete Pipeline

```bash
# Option 1: Full pipeline from scratch
python pipeline.py --stage all --data_path /path/to/dataset

# Option 2: Just knowledge distillation (if teachers exist)
cd knowledge_distillation
python train_sequential.py --config configs/config.yaml

# Option 3: Step by step
python pipeline.py --stage training      # Train base models
python pipeline.py --stage ensemble      # Create ensembles
python pipeline.py --stage conversion    # Convert to ONNX
cd knowledge_distillation
python train_sequential.py               # Train student
python finish_training.py                # Export final models
```

### 8.6 Expected Output Structure

```
knowledge_distillation/
├── checkpoints/
│   ├── checkpoint_after_alexnet.pth
│   ├── checkpoint_after_mobilenet_v2.pth
│   ├── ... (after each teacher)
│   └── class_mapping.json
├── exported_models/
│   ├── student_model.pt
│   ├── student_model.onnx
│   └── student_model.tflite (if conversion successful)
├── metrics/
│   ├── final_evaluation.json
│   └── plots/
│       └── final_confusion_matrix.png
├── logs/
│   └── training_YYYYMMDD_HHMMSS.log
└── training_report.json
```

---

## 9. Troubleshooting

### 9.1 Common Issues

**Issue: CUDA out of memory**
```bash
# Solution: Reduce batch size in config.yaml
dataset:
  batch_size: 8  # Reduce from 16
```

**Issue: Windows multiprocessing error**
```python
# Solution: Use num_workers=0 or add guard
if __name__ == '__main__':
    main()
```

**Issue: Import errors**
```bash
# Solution: Ensure you're in the right directory
cd knowledge_distillation
python train_sequential.py
```

**Issue: TFLite conversion fails**
```bash
# Solution: Use compatible TensorFlow version
pip install tensorflow==2.14.0 onnx-tf==1.10.0
```

**Issue: Model output is dictionary instead of tensor**
```python
# Solution: Extract logits from dict
outputs = model(images)
if isinstance(outputs, dict):
    outputs = outputs['logits']
```

### 9.2 Performance Tips

1. **Use GPU**: Ensure CUDA is available and PyTorch uses it
2. **Reduce workers on Windows**: `num_workers: 4` or lower
3. **Monitor VRAM**: Use `nvidia-smi` to check memory usage
4. **Use mixed precision**: Enable AMP for faster training (if supported)

---

## 10. References

- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network"
- **EWC**: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks"
- **CBAM**: Woo et al., "CBAM: Convolutional Block Attention Module"
- **FPN**: Lin et al., "Feature Pyramid Networks for Object Detection"
- **MobileNetV2**: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks"

---

## License

MIT License - See [LICENSE](../LICENSE) for details.

## Authors

- **Intelli-PEST Team** - [SERVER-246](https://github.com/SERVER-246)

---

**Last Updated:** December 2024  
**Pipeline Version:** 2.0.0
