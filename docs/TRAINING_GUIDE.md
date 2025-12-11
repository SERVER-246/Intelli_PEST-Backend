# Training Guide

## Overview

This guide explains how to train models using the Intelli_PEST-Backend pipeline.

## Quick Start

### 1. Prepare Your Dataset

```
data/
├── train/
│   ├── army_worm/
│   ├── internode_borer/
│   └── ...
├── val/
│   └── [same structure]
└── test/
    └── [same structure]
```

### 2. Run Training

```python
from src.training.base_training import run_optimized_pipeline

# Run complete pipeline
results = run_optimized_pipeline()
```

### 3. Monitor Progress

Check logs in `outputs/logs/training.log` or terminal output.

## Training Configuration

Edit `configs/training_config.yaml` to customize:

### Batch Size & Learning Rate
```yaml
training:
  batch_size: 32
  learning_rate: 0.001
```

### Epochs
```yaml
training:
  epochs_head: 40        # Classifier head training
  epochs_finetune: 25    # Full model fine-tuning
```

### Data Augmentation
```yaml
augmentation:
  crop_scale_min: 0.85
  horizontal_flip: 0.5
  color_jitter_brightness: 0.1
```

## Model Selection

Configure which models to train in `configs/model_config.yaml`:

```yaml
models:
  backbones:
    - alexnet
    - resnet50
    - inception_v3
    - mobilenet_v2
    - efficientnet_b0
    - darknet53
    - yolo11n-cls
```

## K-Fold Cross-Validation

Enables robust model evaluation:

```python
k_fold_cross_validation_optimized(
    backbone_name='resnet50',
    full_dataset=dataset,
    k_folds=5
)
```

## Ensemble Training

### Individual Ensembles

```python
ensemble, acc, metrics = train_single_ensemble(
    trained_models=models_dict,
    train_loader=train_loader,
    val_loader=val_loader,
    fusion_type='attention'  # or 'concat', 'cross'
)
```

### Super Ensemble

Combines all three fusion strategies:

```python
super_ensemble, acc, metrics = train_super_ensemble(
    attention_ensemble=ensemble1,
    concat_ensemble=ensemble2,
    cross_ensemble=ensemble3,
    train_loader=train_loader,
    val_loader=val_loader
)
```

## Monitoring Training

### Real-Time Metrics

Monitor in terminal:
```
HEAD Epoch  1/40 | Train Loss: 2.3456 Acc: 0.4567 | Val Loss: 2.1234 Acc: 0.5234
HEAD Epoch  2/40 | Train Loss: 2.1234 Acc: 0.5234 | Val Loss: 1.9876 Acc: 0.6123
...
```

### Saved Outputs

All results saved in `outputs/`:
- `metrics_output/` - JSON metrics for each model
- `plots_metrics/` - Visualization plots
- `checkpoints/` - Model checkpoints

## Performance Tips

### Speed Up Training

1. **Increase Batch Size** (if VRAM allows)
   ```yaml
   training:
     batch_size: 64  # From 32
   ```

2. **Reduce Epochs**
   ```yaml
   training:
     epochs_head: 20
     epochs_finetune: 10
   ```

3. **Decrease K-Folds**
   ```python
   k_folds=3  # From 5
   ```

### Improve Accuracy

1. **Increase Epochs**
2. **Fine-tune Learning Rate**
3. **Use All Augmentations**
4. **Add More Training Data**

## Troubleshooting

### Out of Memory
```yaml
training:
  batch_size: 16  # Reduce batch size
```

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Increase num_workers in training script
- Use float16 precision if supported

### Poor Accuracy
- Verify dataset is properly split
- Check data augmentation parameters
- Try different learning rates
- Increase training epochs

## Next Steps

- [Conversion Guide](CONVERSION_GUIDE.md) - Convert models to ONNX/TFLite
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Deploy models
