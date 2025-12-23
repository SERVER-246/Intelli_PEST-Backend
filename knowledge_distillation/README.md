# Knowledge Distillation for Pest Classification

Advanced **Sequential Multi-Teacher Knowledge Distillation** pipeline to create a powerful, unified student model for pest classification. The student model learns sequentially from **all 11 teacher models**, one at a time, absorbing each teacher's unique knowledge while preventing catastrophic forgetting using **Elastic Weight Consolidation (EWC)**.

## 🌟 Key Features

- **Sequential Learning**: Train with each teacher one-by-one (not simultaneously)
- **Elastic Weight Consolidation (EWC)**: Prevents forgetting previous teachers' knowledge
- **Multi-Format Teacher Support**: Load teachers from `.pt`, `.pth`, `.onnx`, `.tflite`
- **Enhanced Student Architecture**: 
  - Multi-scale convolution (Inception-style)
  - CBAM attention mechanism
  - Feature Pyramid Network (FPN)
  - Knowledge Consolidation Blocks
- **Comprehensive Export**: PyTorch, ONNX, and TFLite formats
- **Android-Compatible**: TFLite models work with TensorFlow Lite 2.14+

## 📁 Project Structure

```
knowledge_distillation/
├── configs/
│   ├── config.yaml              # Main configuration file
│   ├── training_config.yaml     # Training-specific settings
│   └── class_mapping.json       # Class name to index mapping
├── src/
│   ├── __init__.py
│   ├── enhanced_student_model.py # Advanced student architecture
│   ├── teacher_loader.py         # Multi-format teacher loader
│   ├── sequential_trainer.py     # Sequential distillation with EWC
│   ├── dataset.py                # Data loading and augmentation
│   ├── exporter.py               # Model export utilities
│   └── evaluator.py              # Model evaluation
├── checkpoints/                  # Training checkpoints
├── logs/                         # Training logs
├── metrics/                      # Training metrics (JSON)
├── exported_models/              # Final exported models
├── train.py                      # Standard training script
├── train_sequential.py           # Sequential training script (recommended)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Training

Edit `configs/config.yaml` to set:
- Dataset path
- Teacher model locations (supports .pt, .pth, .onnx, .tflite)
- Training hyperparameters (epochs per teacher, final epochs)
- Export settings

### 3. Run Sequential Training (Recommended)

```bash
# Train sequentially from all 11 teachers
python train_sequential.py --config configs/config.yaml

# With custom output directory
python train_sequential.py --config configs/config.yaml --output-dir ./output

# Resume from checkpoint
python train_sequential.py --config configs/config.yaml --resume checkpoints/student_after_efficientnet_b0.pt
```

### 4. Standard Training (Legacy)

```bash
# Traditional multi-teacher training
python train.py --config configs/config.yaml
```

### 5. Export Only

```bash
python train_sequential.py --config configs/config.yaml --export-only checkpoints/best_student.pt
```

## 🎯 Training Process

### Sequential Learning Phases

The training proceeds through **11 teacher phases** + **1 final refinement phase**:

```
Phase 1/11:  Learning from AlexNet         (20 epochs)
Phase 2/11:  Learning from MobileNet_V2    (20 epochs)
Phase 3/11:  Learning from EfficientNet_B0 (20 epochs)
Phase 4/11:  Learning from ResNet50        (20 epochs)
Phase 5/11:  Learning from DarkNet53       (20 epochs)
Phase 6/11:  Learning from Inception_V3    (20 epochs)
Phase 7/11:  Learning from YOLO11n-cls     (20 epochs)
Phase 8/11:  Learning from Ensemble_Attn   (20 epochs)
Phase 9/11:  Learning from Ensemble_Concat (20 epochs)
Phase 10/11: Learning from Ensemble_Cross  (20 epochs)
Phase 11/11: Learning from Super_Ensemble  (20 epochs)
Final Phase: Ensemble Refinement           (30 epochs)
─────────────────────────────────────────────────────
Total:       250 epochs
```

### Elastic Weight Consolidation (EWC)

After each teacher phase, the trainer:
1. Computes Fisher Information Matrix for important parameters
2. Stores optimal weights from that teacher
3. Adds EWC penalty during subsequent training to prevent forgetting

This ensures the student retains knowledge from ALL teachers.

## 📊 Dataset Format

The dataset should be organized in ImageFolder format:
```
IMAGE DATASET/
├── Healthy/
│   ├── image1.jpg
│   └── ...
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

**Current Dataset Stats:**
- 11 classes
- 3,067 training samples
- 773 validation samples
- 80/20 train/val split

## 🎓 Teacher Models (11 Total)

All teachers are pre-trained ONNX models located in `tflite_models_compatible/onnx_models/`:

| Model | Type | Weight | Description |
|-------|------|--------|-------------|
| AlexNet | Base | 0.8 | Classic CNN architecture |
| MobileNet_V2 | Base | 1.0 | Efficient mobile architecture |
| EfficientNet_B0 | Base | 1.2 | Compound scaling |
| ResNet50 | Base | 1.0 | Residual connections |
| DarkNet53 | Base | 1.0 | YOLO backbone |
| Inception_V3 | Base | 1.0 | Multi-scale features |
| YOLO11n-cls | Base | 1.0 | Ultralytics classifier |
| Ensemble_Attention | Ensemble | 1.5 | Attention-based fusion |
| Ensemble_Concat | Ensemble | 1.5 | Concatenation fusion |
| Ensemble_Cross | Ensemble | 1.5 | Cross-attention fusion |
| Super_Ensemble | Ensemble | 2.0 | Meta-ensemble (highest weight) |

### Multi-Format Support

The teacher loader supports multiple formats:
```python
from src.teacher_loader import MultiFormatTeacherEnsemble

# Automatically detects format from extension
teachers = MultiFormatTeacherEnsemble(teacher_configs, device='cuda')

# Supported formats:
# - .pt, .pth  → PyTorch (GPU accelerated)
# - .onnx      → ONNX Runtime (CPU/GPU)
# - .tflite    → TensorFlow Lite (CPU)
```

## 📈 Training Features

- **Sequential Distillation**: One teacher at a time for focused learning
- **Elastic Weight Consolidation**: Prevents catastrophic forgetting
- **Temperature Scaling**: Soften probability distributions (T=4.0)
- **Weighted Ensemble**: Higher weight for ensemble teachers
- **Mixed Precision**: FP16 training for faster computation (optional)
- **Data Augmentation**: Rotation, flip, color jitter, random erasing
- **Weighted Sampling**: Handle class imbalance automatically
- **Comprehensive Logging**: Track all metrics per teacher phase
- **Checkpoint Saving**: Save after each teacher for resumability

## 🔧 Enhanced Student Model Architecture

The enhanced student model (`src/enhanced_student_model.py`) features:

### Core Components

1. **Multi-Scale Convolution Block** (Inception-style)
   - 1x1, 3x3, 5x5 parallel convolutions
   - Captures features at multiple scales

2. **CBAM Attention** (Channel + Spatial)
   - Channel attention: "what" to focus on
   - Spatial attention: "where" to focus

3. **Inverted Residual Blocks** (MobileNet-style)
   - Expansion → Depthwise → Projection
   - Efficient parameter usage

4. **Feature Pyramid Network (FPN)**
   - Multi-scale feature fusion
   - Top-down pathway with lateral connections

5. **Knowledge Consolidation Blocks**
   - Self-attention for feature integration
   - Helps absorb diverse teacher knowledge

### Model Variants

| Variant | Size | Parameters | Use Case |
|---------|------|------------|----------|
| Large | ~92 MB | ~24M | Maximum accuracy |
| Medium | ~46 MB | ~12M | Balanced (default) |
| Small | ~23 MB | ~6M | Mobile deployment |

### Current Configuration
```yaml
student:
  size: medium
  base_channels: 48
  expand_ratio: 4
  num_classes: 11
```

## 📦 Export Formats

After training, models are exported to three formats:

| Format | File | Use Case |
|--------|------|----------|
| PyTorch | `final_student.pt` | Training, fine-tuning |
| ONNX | `final_student.onnx` | Cross-platform inference |
| TFLite | `final_student.tflite` | Android/mobile deployment |

### Android Compatibility

TFLite models are exported with:
- TensorFlow 2.14+ compatibility
- Proper input/output specifications
- Quantization-ready (optional)

## 📋 Metrics Tracked

### Per-Teacher Metrics
- Training/Validation Loss
- Accuracy per epoch
- Best accuracy achieved
- EWC loss contribution

### Final Metrics
- Overall Accuracy
- Per-class Accuracy
- Precision, Recall, F1 Score
- Confusion Matrix
- Teacher comparison (student vs each teacher)

## 📄 Output Files

After training completes:

```
checkpoints/
├── student_after_alexnet.pt
├── student_after_mobilenet_v2.pt
├── student_after_efficientnet_b0.pt
├── ... (after each teacher)
├── student_after_super_ensemble.pt
├── student_final.pt              # Final refined model
└── class_mapping.json

exported_models/
├── final_student.pt
├── final_student.onnx
└── final_student.tflite

metrics/
├── sequential_training_report.json
├── teacher_metrics.json
└── final_evaluation.json

logs/
└── training_YYYYMMDD_HHMMSS.log
```

## 🔬 Configuration Options

See `configs/config.yaml` for all options:

```yaml
# Dataset Configuration
dataset:
  path: "G:/AI work/IMAGE DATASET"
  image_size: 256
  batch_size: 16
  train_split: 0.8
  num_workers: 4

# Student Model Configuration
student:
  num_classes: 11
  size: "medium"        # small, medium, large
  base_channels: 48
  expand_ratio: 4

# Teacher Models Configuration
teachers:
  base_path: "path/to/teacher/models"
  models:
    - name: "alexnet"
      file: "alexnet_pest.onnx"
      weight: 0.8
    - name: "super_ensemble"
      file: "super_ensemble_pest.onnx"
      weight: 2.0
    # ... more teachers

# Training Configuration
training:
  epochs_per_teacher: 20    # Epochs to train with each teacher
  final_ensemble_epochs: 30 # Final refinement epochs
  learning_rate: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
  ewc_lambda: 1000          # EWC regularization strength

# Distillation Configuration
distillation:
  temperature: 4.0          # Soft label temperature
  alpha: 0.7                # Soft label weight
  beta: 0.3                 # Hard label weight
```

## 🖥️ Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (tested on RTX 4500 Ada 25.8GB)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and checkpoints

## ⚡ Performance Tips

1. **Use GPU for student**: Ensure CUDA is available
2. **Reduce batch size** if OOM: Start with 16, increase if stable
3. **Use fewer workers** on Windows: `num_workers: 4` or lower
4. **Mixed precision**: Enable for 2x faster training (if supported)

## 📝 License

MIT License

## 👥 Authors

Knowledge Distillation Pipeline - December 2024

## 🔄 Changelog

### v2.0.0 (December 2024)
- **NEW**: Sequential training from all 11 teachers
- **NEW**: Elastic Weight Consolidation (EWC) to prevent forgetting
- **NEW**: Multi-format teacher loader (.pt, .pth, .onnx, .tflite)
- **NEW**: Enhanced student model with CBAM, FPN, Knowledge Consolidation
- **IMPROVED**: Better logging and checkpoint management
- **FIXED**: Dataset function naming (`create_dataloaders`)
- **FIXED**: Class alignment for "porcupine damage" folder

### v1.0.0 (December 2024)
- Initial release with multi-teacher distillation
- Basic student model
- ONNX teacher support only
