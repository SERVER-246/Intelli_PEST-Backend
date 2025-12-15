# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- CUDA 11.0+ (optional, for GPU acceleration)

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Intelli_PEST-Backend.git
cd Intelli_PEST-Backend
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Package

**Option A: Development Installation (Recommended)**
```bash
pip install -e .
```

**Option B: Requirements File**
```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```python
python -c "
from src.training import base_training
from src.conversion import onnx_converter
print('Γ£ô Installation successful!')
"
```

## Configuration

### 1. Update Dataset Paths

Edit the training scripts to point to your dataset:

```python
RAW_DIR = Path(r"D:\\ML-Model Code\\pest_dataset")
SPLIT_DIR = Path(r"D:\\ML-Model Code\\split_dataset")
```

### 2. Configure Training Parameters

Edit `configs/training_config.yaml`:

```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  epochs_head: 40
```

### 3. Set Output Directories

Create required directories:

```bash
mkdir data
mkdir outputs
mkdir logs
```

## Troubleshooting

### ImportError: No module named 'torch'

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CUDA Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
export CUDA_VISIBLE_DEVICES=""  # Linux/Mac
set CUDA_VISIBLE_DEVICES=  # Windows
```

### Memory Issues

Reduce batch size in `configs/training_config.yaml`:

```yaml
training:
  batch_size: 16  # Reduced from 32
```

## Next Steps

- Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training instructions
- See [CONVERSION_GUIDE.md](CONVERSION_GUIDE.md) for model conversion
- Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment options

## Support

For issues, please open an issue on GitHub or check the troubleshooting section.
