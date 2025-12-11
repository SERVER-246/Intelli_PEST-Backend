# Getting Started with Intelli_PEST-Backend

## 🚀 Quick Start Guide (5 Minutes)

This guide will help you clone and run the entire pipeline from scratch.

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.8 or higher** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **8GB+ RAM** (16GB recommended for training)
- **GPU with CUDA support** (optional but highly recommended)
- **10GB free disk space**

---

## Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/SERVER-246/Intelli_PEST-Backend.git
cd Intelli_PEST-Backend
```

---

## Step 2: Set Up Python Environment

### Option A: Using Virtual Environment (Recommended)

**On Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**On Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Install as Package

```bash
# Install in development mode
pip install -e .
```

---

## Step 3: Prepare Your Dataset

Organize your dataset in the following structure:

```
data/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── class3/
    ├── image1.jpg
    └── ...
```

### Example for Pest Detection:

```
data/
├── army_worm/
│   └── *.jpg
├── internode_borer/
│   └── *.jpg
├── pink_borer/
│   └── *.jpg
├── stalk_borer/
│   └── *.jpg
└── termite/
    └── *.jpg
```

**Update the dataset path** in `configs/training_config.yaml`:

```yaml
data:
  data_folder: "data/"  # Change this to your dataset path
  image_size: 256
```

---

## Step 4: Run Training (3 Options)

### Option A: Train Individual Backbone Model

```python
# Run from project root
python -c "from src.training.base_training import run_optimized_pipeline; run_optimized_pipeline()"
```

Or create a simple script `train.py`:
```python
from src.training.base_training import run_optimized_pipeline

if __name__ == "__main__":
    results = run_optimized_pipeline()
    print(f"Training completed! Results: {results}")
```

Then run:
```bash
python train.py
```

### Option B: Train Ensemble Models

```python
from src.training.ensemble_training import train_single_ensemble

# Train attention-based ensemble
train_single_ensemble(fusion_type='attention')
```

### Option C: Use Configuration Files

Edit `configs/training_config.yaml` to customize:
- Batch size
- Learning rate
- Number of epochs
- Data augmentation

Then run your training script.

---

## Step 5: Convert Models to TFLite

### Quick Conversion Script

```bash
# Run the interactive conversion tool
python scripts/quick_start.py
```

Or use the batch script:

**On Windows:**
```powershell
.\scripts\run_tflite_conversion.bat
```

**On Linux/Mac:**
```bash
bash scripts/run_tflite_conversion.sh
```

### Manual Conversion (Python)

```python
from src.conversion.tflite_converter import ONNXToTFLiteConverter

converter = ONNXToTFLiteConverter(
    onnx_models_dir='Base-dir/deployment_models/',
    output_dir='outputs/tflite_models/'
)

# Convert all models
converter.convert_all_models()
```

---

## Step 6: Verify & Test

### Check Installation

```python
# test_installation.py
import torch
import tensorflow as tf
import onnx

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"TensorFlow version: {tf.__version__}")
print(f"ONNX version: {onnx.__version__}")

print("\n✅ All dependencies installed successfully!")
```

Run:
```bash
python test_installation.py
```

### Run Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_training.py -v
```

---

## Step 7: Monitor Training Progress

Training outputs will be saved to:

```
outputs/
├── checkpoints/          # Model checkpoints (.pth files)
├── logs/                 # Training logs
├── plots/                # Confusion matrices, ROC curves
└── metrics/              # Training metrics (CSV)
```

### View Training Logs

```bash
# Tail the training log
tail -f outputs/logs/training.log

# On Windows (PowerShell)
Get-Content outputs/logs/training.log -Wait -Tail 50
```

---

## Common Workflows

### Workflow 1: Train → Evaluate → Deploy

```python
# 1. Train base model
from src.training.base_training import run_optimized_pipeline
results = run_optimized_pipeline()

# 2. Export to ONNX (already done during training)
# Models are saved to Base-dir/deployment_models/

# 3. Convert to TFLite
from src.conversion.tflite_converter import ONNXToTFLiteConverter
converter = ONNXToTFLiteConverter(
    onnx_models_dir='Base-dir/deployment_models/',
    output_dir='outputs/tflite_models/'
)
converter.convert_all_models()

# 4. Validate accuracy
from src.conversion.model_validator import ModelAccuracyValidator
validator = ModelAccuracyValidator()
validator.validate_all_models()
```

### Workflow 2: Quick TFLite Conversion Only

If you already have trained models:

```bash
# Interactive menu
python scripts/quick_start.py

# Select option 1: Convert All Models
```

### Workflow 3: Resume Training

```python
from src.training.base_training import train_backbone_optimized

# Load checkpoint and continue training
results = train_backbone_optimized(
    backbone_name='resnet50',
    resume_checkpoint='outputs/checkpoints/resnet50_best.pth'
)
```

---

## Configuration Guide

### Training Configuration (`configs/training_config.yaml`)

Key parameters to adjust:

```yaml
training:
  batch_size: 32          # Reduce if OOM error (16, 8)
  epochs_head: 40         # Epochs for head training
  epochs_finetune: 25     # Epochs for fine-tuning
  learning_rate: 0.001    # Initial learning rate

data:
  data_folder: "data/"    # Your dataset path
  train_split: 0.8        # 80% for training
  val_split: 0.1          # 10% for validation
  test_split: 0.1         # 10% for testing

performance:
  num_workers: 4          # Data loader workers
  pin_memory: true        # Use CUDA pinned memory
```

### Model Configuration (`configs/model_config.yaml`)

Select backbones to train:

```yaml
backbones:
  - alexnet
  - resnet50
  - mobilenet_v2          # Lightweight for mobile
  - efficientnet_b0       # Good accuracy/speed tradeoff
  # - inception_v3        # Uncomment to enable
  # - darknet53
  # - yolo11n-cls
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
- Reduce `batch_size` in `configs/training_config.yaml`
- Use smaller models (MobileNetV2, EfficientNetB0)
- Close other GPU applications

```yaml
training:
  batch_size: 16  # or 8
```

### Issue: Import Errors

**Solution:**
```bash
# Make sure you're in the project root
cd Intelli_PEST-Backend

# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)"              # Windows PowerShell
```

### Issue: Slow Training

**Solutions:**
- Enable CUDA: `device: "cuda"` in config
- Increase `num_workers` in config
- Use mixed precision training (enabled by default)
- Use smaller `image_size`: 224 instead of 256

### Issue: TFLite Conversion Fails

**Solution:**
```bash
# Reinstall TensorFlow and ONNX
pip install --upgrade tensorflow onnx onnx-tf

# Check ONNX model validity first
python -c "import onnx; model = onnx.load('model.onnx'); onnx.checker.check_model(model)"
```

---

## Performance Benchmarks

Expected training times (on NVIDIA RTX 3080):

| Model          | Training Time | Accuracy | Size (PyTorch) | Size (TFLite) |
|----------------|---------------|----------|----------------|---------------|
| AlexNet        | ~30 min       | 89-92%   | 233 MB         | 58 MB         |
| ResNet50       | ~45 min       | 93-95%   | 98 MB          | 25 MB         |
| MobileNetV2    | ~25 min       | 91-93%   | 14 MB          | 3.5 MB        |
| EfficientNetB0 | ~40 min       | 94-96%   | 21 MB          | 5 MB          |
| Ensemble       | ~2 hours      | 96-98%   | 500 MB         | 125 MB        |

*Times vary based on dataset size and hardware*

---

## Next Steps

1. **Explore Documentation**
   - [Installation Guide](docs/INSTALLATION.md)
   - [Training Guide](docs/TRAINING_GUIDE.md)
   - [Conversion Guide](docs/CONVERSION_GUIDE.md)
   - [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)

2. **Customize Configuration**
   - Edit `configs/*.yaml` files
   - Adjust hyperparameters for your dataset

3. **Deploy Models**
   - See [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
   - Options: PyTorch, ONNX, TFLite

4. **Contribute**
   - Report issues on GitHub
   - Submit pull requests
   - Share your results

---

## Support & Resources

- **Documentation**: See `docs/` folder
- **Issues**: Report on GitHub Issues
- **Discussions**: GitHub Discussions
- **Examples**: Check `examples/` folder (if available)

---

## Quick Command Reference

```bash
# Setup
git clone https://github.com/SERVER-246/Intelli_PEST-Backend.git
cd Intelli_PEST-Backend
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train
python train.py

# Convert to TFLite
python scripts/quick_start.py

# Test
python -m pytest tests/ -v

# Check dataset
python -m src.utils.data_counter

# View logs
tail -f outputs/logs/training.log
```

---

## Complete Example Script

Create `run_full_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Complete pipeline: Train → Convert → Validate
"""

from src.training.base_training import run_optimized_pipeline
from src.conversion.tflite_converter import ONNXToTFLiteConverter
from src.conversion.model_validator import ModelAccuracyValidator

def main():
    print("=" * 80)
    print("INTELLI_PEST COMPLETE PIPELINE")
    print("=" * 80)
    
    # Step 1: Train models
    print("\n[1/3] Training models...")
    results = run_optimized_pipeline()
    print(f"✓ Training complete: {results}")
    
    # Step 2: Convert to TFLite
    print("\n[2/3] Converting to TFLite...")
    converter = ONNXToTFLiteConverter(
        onnx_models_dir='Base-dir/deployment_models/',
        output_dir='outputs/tflite_models/'
    )
    converter.convert_all_models()
    print("✓ Conversion complete")
    
    # Step 3: Validate accuracy
    print("\n[3/3] Validating accuracy...")
    validator = ModelAccuracyValidator()
    validator.validate_all_models()
    print("✓ Validation complete")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

Run it:
```bash
python run_full_pipeline.py
```

---

## That's It! 🎉

You're now ready to use the Intelli_PEST-Backend pipeline. If you encounter any issues, check the troubleshooting section or refer to the detailed documentation in the `docs/` folder.

Happy coding! 🐛🔬
