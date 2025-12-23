# Knowledge Distillation Module

A multi-teacher knowledge distillation pipeline for training lightweight pest classification models.

## Overview

This module trains a compact student model (~5 MB) using knowledge from 11 pre-trained teacher models, achieving near-teacher performance with significantly reduced model size.

### Features

- **Multi-Teacher Knowledge Distillation**: Learns from 11 diverse teacher architectures
- **Custom CNN Student**: Efficient depthwise separable convolutions (~5 MB, 1.3M parameters)
- **Soft Label Learning**: Temperature-scaled soft targets from teacher ensemble
- **Comprehensive Evaluation**: Confusion matrices, per-class metrics, training curves
- **Multi-Format Export**: PyTorch, ONNX (opset 13), TFLite (TF 2.14 compatible)

## Teacher Models

| Model | Type | Weight |
|-------|------|--------|
| MobileNetV2 | Base | 1.0 |
| ResNet50 | Base | 1.0 |
| InceptionV3 | Base | 1.0 |
| EfficientNet-B0 | Base | 1.0 |
| DarkNet53 | Base | 1.0 |
| AlexNet | Base | 1.0 |
| YOLO11n-cls | Base | 1.0 |
| Ensemble-Attention | Ensemble | 1.5 |
| Ensemble-Concat | Ensemble | 1.5 |
| Ensemble-Cross | Ensemble | 1.5 |
| Super-Ensemble | Ensemble | 2.0 |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Dataset Path

Edit `configs/config.yaml`:

```yaml
dataset:
  path: "/path/to/your/dataset"  # ImageFolder format required
```

Dataset should be organized as:
```
dataset/
├── class_1/
│   ├── image1.jpg
│   └── image2.jpg
├── class_2/
│   └── ...
└── class_n/
```

### 3. Start Training

```bash
cd knowledge_distillation
python train.py --config configs/config.yaml
```

### 4. Monitor Progress

Training logs are saved to `logs/` directory with:
- Training/validation metrics per epoch
- Best model checkpoints
- Evaluation plots and confusion matrices

## Output Structure

After training completes:

```
knowledge_distillation/
├── checkpoints/
│   ├── best_checkpoint.pt      # Best validation accuracy
│   └── latest_checkpoint.pt    # Most recent
├── models/
│   └── student/
│       ├── pytorch/
│       │   └── student_model.pt
│       ├── onnx/
│       │   └── student_model.onnx
│       └── tflite/
│           └── student_model.tflite
├── metrics/
│   ├── classification_report.json
│   └── evaluation_summary.json
├── plots/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── per_class_accuracy.png
└── logs/
    └── training_YYYYMMDD_HHMMSS.log
```

## Student Model Architecture

```
StudentCNN (Custom Lightweight CNN)
├── Input: 256x256x3
├── Conv Block 1: 3→32 channels (depthwise separable)
├── Conv Block 2: 32→64 channels
├── Conv Block 3: 64→128 channels
├── Conv Block 4: 128→256 channels
├── Conv Block 5: 256→512 channels
├── Global Average Pooling
├── Dropout (0.3)
└── Dense: 512→12 classes

Total Parameters: ~1.3M
Model Size: ~5 MB
```

## Training Configuration

Key hyperparameters in `configs/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Training epochs |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 0.001 | Initial LR |
| `temperature` | 4.0 | KD temperature |
| `alpha` | 0.7 | Soft label weight |
| `beta` | 0.3 | Hard label weight |

## Command Line Options

```bash
python train.py [OPTIONS]

Options:
  --config PATH       Config file path (default: configs/config.yaml)
  --data_dir PATH     Override dataset directory
  --epochs INT        Override number of epochs
  --batch_size INT    Override batch size
  --lr FLOAT          Override learning rate
  --output_dir PATH   Override output directory
  --model_type STR    Student model type: standard|small|tiny
  --resume PATH       Resume from checkpoint
  --no_cuda           Disable CUDA
  --export_only PATH  Export from checkpoint without training
```

## Export Formats

### PyTorch (.pt)
```python
import torch
from src.student_model import create_student_model

model = create_student_model(num_classes=12)
checkpoint = torch.load('models/student/pytorch/student_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### ONNX
```python
import onnxruntime as ort

session = ort.InferenceSession('models/student/onnx/student_model.onnx')
output = session.run(None, {'input': image_tensor})
```

### TFLite (Android Compatible)
```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='models/student/tflite/student_model.tflite')
interpreter.allocate_tensors()
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Model Size | < 15 MB | Achieved: ~5 MB |
| Accuracy | > 85% | Teacher-dependent |
| Inference | < 50ms | On mobile devices |

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in config:
```yaml
dataset:
  batch_size: 16  # or 8
```

### ONNX Runtime CPU Only
Install GPU version:
```bash
pip install onnxruntime-gpu
```

### TFLite Conversion Fails
Ensure TensorFlow 2.14-2.16 is installed:
```bash
pip install tensorflow==2.14.0
```

## License

MIT License - See LICENSE file for details.
