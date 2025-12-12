# TFLite Conversion Guide

This guide details the process for converting PyTorch models to TensorFlow Lite (TFLite) using the specialized environment and scripts in this repository.

## Environment Setup

The conversion process requires specific versions of PyTorch, TensorFlow, and ONNX to work correctly on Windows. We have created a dedicated environment for this.

### 1. Create the Environment

```powershell
# From the root of the repository
python -m venv venv_tflite
```

### 2. Activate the Environment

```powershell
# Windows
.\venv_tflite\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements_tflite.txt
```

**Key Dependencies:**
- `torch==2.3.1`
- `torchvision==0.18.1`
- `tensorflow==2.16.1` (or compatible 2.x)
- `onnx==1.16.0`
- `onnx2tf`
- `numpy==1.26.4`

## Running the Conversion

The main script is located at `src/conversion/pytorch_to_tflite_quantized.py`.

### Command

```powershell
python src/conversion/pytorch_to_tflite_quantized.py
```

### What it does

1.  **Scans** for `.pt` models in the configured directory (default: `D:\Base-dir\deployment_models`).
2.  **Iterates** through each model:
    -   **Export to ONNX**: Uses `torch.onnx.export` with `opset_version=17`.
    -   **Convert to SavedModel**: Uses `onnx2tf` to generate a TensorFlow SavedModel.
    -   **Convert to TFLite**: Uses `tf.lite.TFLiteConverter` with `optimizations=[tf.lite.Optimize.DEFAULT]` (Dynamic Range Quantization).
3.  **Verifies** the output TFLite model by loading it and checking input/output shapes.
4.  **Saves** the result in `tflite_models/`.

### Process Isolation

The script uses Python's `subprocess` module to run each model conversion in a separate process. This is crucial for:
-   **Memory Management**: Releasing TensorFlow/CUDA memory after each model.
-   **Stability**: Preventing a crash in one model (e.g., `bad allocation`) from stopping the entire batch.

## Troubleshooting

### `bad allocation` Error
If you see this error, the conversion process ran out of RAM.
-   **Solution**: Close other applications or try converting the large model on a machine with more RAM (16GB+).

### `adaptive_avg_pool2d` Error
This occurs during ONNX export for models using Adaptive Average Pooling with dynamic input sizes.
-   **Solution**: The model architecture needs to be modified to use fixed-size pooling or fixed input sizes before export.

### Network Timeouts
The `onnx2tf` tool tries to download test images. The script includes a "monkey patch" to bypass this, but ensure your firewall isn't blocking local loopback or basic python networking.
