# TFLite Conversion Results

## Environment
- **Python**: 3.10
- **PyTorch**: 2.3.1
- **TensorFlow**: 2.16.1
- **ONNX**: 1.16.0
- **Date**: December 12, 2025

## Conversion Summary

Successfully converted **1 out of 11** models from PyTorch to TFLite using Dynamic Range Quantization.

### ✅ Successfully Converted

| Model | PyTorch Size | TFLite Size | Compression | Status |
|-------|--------------|-------------|-------------|--------|
| mobilenet_v2 | 12.17 MB | 3.18 MB | 73.9% | ✅ Success |

### ❌ Failed Conversions

| Model | Size | Error | Reason |
|-------|------|-------|--------|
| darknet53 | 81.28 MB | `bad allocation` | Out of memory during conversion (requires 16GB+ RAM) |
| ensemble_attention | 577.58 MB | ONNX export error | `adaptive_avg_pool2d` operator not supported in ONNX export |
| ensemble_concat | 579.58 MB | ONNX export error | `adaptive_avg_pool2d` operator not supported in ONNX export |
| ensemble_cross | 621.65 MB | ONNX export error | `adaptive_avg_pool2d` operator not supported in ONNX export |
| super_ensemble | ~1000 MB | ONNX export error | `adaptive_avg_pool2d` operator not supported in ONNX export |
| alexnet | - | Unknown | Pending conversion |
| resnet50 | - | Unknown | Pending conversion |
| inception_v3 | - | Unknown | Pending conversion |
| efficientnet_b0 | - | Unknown | Pending conversion |
| yolo11n-cls | - | Unknown | Pending conversion |

## Known Issues

### 1. Memory Limitations
**Problem**: Large models (>80MB) fail with `bad allocation` error.  
**Solution**: Run conversion on a machine with 16GB+ RAM or use cloud GPU instances.

### 2. Adaptive Pooling ONNX Export
**Problem**: Models using `nn.AdaptiveAvgPool2d` fail during ONNX export.  
**Error**: `Unsupported: ONNX export of operator adaptive_avg_pool2d, output size that are not factor of input size`

**Solution**: Modify the PyTorch model architecture to:
- Replace `nn.AdaptiveAvgPool2d(6)` with `nn.AvgPool2d(kernel_size=X, stride=Y)` with fixed sizes
- OR use `torch.nn.functional.adaptive_avg_pool2d` with compatible output sizes
- OR fix the input image size to make adaptive pooling deterministic

## Recommendations

1. **For Mobile Deployment**: Use the successfully converted `mobilenet_v2.tflite` model (3.18 MB) - it's optimized for mobile devices.

2. **For Ensemble Models**: Consider:
   - Using ONNX Runtime directly in your Android app instead of TFLite
   - Retraining ensemble models with fixed pooling layers
   - Using model distillation to compress ensembles into a single smaller model

3. **For Large Models**: 
   - Convert on a cloud instance with more RAM
   - Use quantization-aware training to get smaller base models
   - Apply pruning before conversion

## Next Steps

To convert the remaining models:
1. Increase system RAM to 16GB+
2. Modify ensemble model architectures to remove adaptive pooling
3. Re-run: `python src/conversion/pytorch_to_tflite_quantized.py`
