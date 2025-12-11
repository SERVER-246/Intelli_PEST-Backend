# Model Conversion Guide

## Overview

This guide explains how to convert models to ONNX and TensorFlow Lite formats.

## Conversion Pipeline

```
PyTorch Model → ONNX → TensorFlow Lite
```

## ONNX Conversion

### Using the Converter Script

```python
from src.conversion.onnx_converter import export_model_to_onnx

onnx_path = export_model_to_onnx(
    model=trained_model,
    model_name='resnet50',
    input_size=256,
    save_dir='outputs/deployment_models'
)
```

### Configuration

Edit `configs/conversion_config.yaml`:

```yaml
onnx:
  opset_version: 13
  do_constant_folding: true
```

## TensorFlow Lite Conversion

### Quick Start Script

```bash
# Windows
scripts/run_tflite_conversion.bat

# Linux/Mac
python scripts/quick_start.py
```

### Programmatic Usage

```python
from src.conversion.tflite_converter import ONNXToTFLiteConverter

converter = ONNXToTFLiteConverter(
    onnx_models_dir='outputs/onnx_models',
    tflite_output_dir='outputs/tflite_models'
)

report = converter.convert_all_models(create_optimized_versions=True)
```

### Optimization Levels

Three versions are created:

1. **Default (Full Precision)**
   - Accuracy: 100%
   - Size: ~100% of original
   - Use: Accuracy-critical applications

2. **Float16 Quantization**
   - Accuracy: 99.9%+
   - Size: ~50% of original
   - Use: Balanced scenarios

3. **Dynamic Range Quantization**
   - Accuracy: 99.5%+
   - Size: ~25% of original
   - Use: Edge devices with memory constraints

## Model Validation

### Accuracy Validation

```python
from src.conversion.model_validator import ModelAccuracyValidator

validator = ModelAccuracyValidator(
    onnx_models_dir='outputs/onnx_models',
    tflite_models_dir='outputs/tflite_models'
)

validation_results = validator.validate_all_models()
```

### Expected Thresholds

- MAE (Mean Absolute Error): < 1e-6
- Cosine Similarity: > 0.9999
- Relative Error: < 0.001

### Handling Validation Failures

If validation fails:

1. Check input preprocessing is identical
2. Verify model quantization settings
3. Try with fewer optimization passes
4. Use more test samples (increase `num_test_samples`)

## Comparison Analysis

### Generate Comparison Report

```python
from src.conversion.comparison_analyzer import ModelComparisonAnalyzer

analyzer = ModelComparisonAnalyzer(
    onnx_models_dir='outputs/onnx_models',
    tflite_models_dir='outputs/tflite_models'
)

report = analyzer.generate_comparison_report()
```

### Output Report Includes

- Model sizes (bytes)
- Size reduction percentage
- Accuracy metrics
- Inference time comparison
- Recommendations

## Configuration Options

### TFLite Options

```yaml
tflite:
  optimization_levels:
    - "default"         # Full precision
    - "float16"         # Float16 quantization
    - "dynamic_range"   # Dynamic range quantization
  
  validation:
    num_test_samples: 10
    mae_threshold: 1e-6
    cosine_similarity_threshold: 0.9999
```

## Batch Conversion

Convert all models at once:

```bash
python scripts/quick_start.py --mode convert_all --output-dir outputs/tflite
```

## Troubleshooting

### Conversion Fails

```python
# Try with reduced optimization
converter.convert_model(
    onnx_path='model.onnx',
    optimization_level='default'  # Skip optimizations
)
```

### Large File Sizes

Use higher optimization levels:

```python
converter.convert_model(
    onnx_path='model.onnx',
    optimization_level='dynamic_range'  # Maximum compression
)
```

### Inference Errors

Verify input/output shapes:

```python
import onnxruntime
sess = onnxruntime.InferenceSession('model.onnx')
print(sess.get_inputs()[0].shape)    # Input shape
print(sess.get_outputs()[0].shape)   # Output shape
```

## Next Steps

- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Deploy converted models
- [Training Guide](TRAINING_GUIDE.md) - Train new models
