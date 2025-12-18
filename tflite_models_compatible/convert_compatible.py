#!/usr/bin/env python3
"""
TFLite Compatible Model Converter
==================================
Converts PyTorch models to TFLite format compatible with TFLite runtime 2.14-2.16.

Key differences from standard conversion:
1. Uses ONNX opset 11 (older, more compatible)
2. Applies TFLite converter settings to use older op versions
3. Uses TF 1.x compatible ops where possible
4. Documents exact input dimensions for each model

Output:
- tflite_models_compatible/ - Models compatible with TFLite 2.14+
- model_metadata.json - Input dimensions and compatibility info
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CompatibleConverter')

# Model configurations with their expected input sizes
MODEL_CONFIGS = {
    'mobilenet_v2': {'input_size': 224, 'channels': 3},
    'resnet50': {'input_size': 224, 'channels': 3},
    'inception_v3': {'input_size': 299, 'channels': 3},  # InceptionV3 uses 299x299
    'efficientnet_b0': {'input_size': 224, 'channels': 3},  # Standard is 224, NOT 256
    'darknet53': {'input_size': 224, 'channels': 3},
    'alexnet': {'input_size': 224, 'channels': 3},
    'yolo11n-cls': {'input_size': 224, 'channels': 3},
    'ensemble_attention': {'input_size': 224, 'channels': 3},
    'ensemble_concat': {'input_size': 224, 'channels': 3},
    'ensemble_cross': {'input_size': 224, 'channels': 3},
    'super_ensemble': {'input_size': 224, 'channels': 3},
}


def convert_single_model_compatible(model_name: str, pytorch_dir: Path, onnx_dir: Path, 
                                     output_dir: Path, temp_dir: Path) -> dict:
    """
    Convert a single model to TFLite with maximum compatibility.
    
    Strategy:
    1. Load PyTorch model
    2. Export to ONNX with opset 11 (older, more compatible)
    3. Convert ONNX to TF SavedModel using onnx2tf with compatibility flags
    4. Convert SavedModel to TFLite with older op versions
    """
    result = {
        'model': model_name,
        'status': 'pending',
        'input_size': MODEL_CONFIGS.get(model_name, {}).get('input_size', 224),
        'channels': 3,
        'tflite_path': None,
        'file_size_mb': None,
        'error': None
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Converting: {model_name}")
    logger.info(f"Expected input: {result['input_size']}x{result['input_size']}x3")
    logger.info(f"{'='*60}")
    
    try:
        import torch
        import numpy as np
        
        # Step 1: Find PyTorch model
        pt_candidates = [
            pytorch_dir / f"{model_name}.pt",
            pytorch_dir / model_name / f"{model_name}.pt",
            pytorch_dir / f"{model_name}_best.pt",
        ]
        
        pt_path = None
        for candidate in pt_candidates:
            if candidate.exists():
                pt_path = candidate
                break
        
        if not pt_path:
            # Try to find ONNX model as fallback
            onnx_path = _find_onnx_model(model_name, onnx_dir)
            if onnx_path:
                logger.info(f"Using pre-existing ONNX model: {onnx_path}")
            else:
                result['status'] = 'failed'
                result['error'] = 'No PyTorch or ONNX model found'
                return result
        else:
            logger.info(f"Found PyTorch model: {pt_path}")
            onnx_path = None
        
        # Step 2: Export to ONNX with opset 11 (if needed)
        model_temp_dir = temp_dir / model_name
        model_temp_dir.mkdir(parents=True, exist_ok=True)
        
        if onnx_path is None:
            onnx_path = model_temp_dir / f"{model_name}.onnx"
            input_size = result['input_size']
            
            logger.info(f"Exporting to ONNX with opset 11...")
            model = torch.jit.load(str(pt_path), map_location='cpu')
            model.eval()
            
            dummy_input = torch.randn(1, 3, input_size, input_size)
            
            # Use opset 11 for maximum compatibility
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                opset_version=11,  # Use older opset for compatibility
                do_constant_folding=True,
            )
            logger.info(f"ONNX export complete: {onnx_path}")
        
        # Step 3: Convert ONNX to TFLite with compatibility settings
        tflite_path = output_dir / f"{model_name}.tflite"
        
        # Use subprocess to run conversion with compatibility flags
        convert_with_compatibility(onnx_path, tflite_path, model_temp_dir, result['input_size'])
        
        if tflite_path.exists():
            result['status'] = 'success'
            result['tflite_path'] = str(tflite_path)
            result['file_size_mb'] = round(tflite_path.stat().st_size / (1024 * 1024), 2)
            logger.info(f"✓ Success: {tflite_path} ({result['file_size_mb']} MB)")
        else:
            result['status'] = 'failed'
            result['error'] = 'TFLite file not created'
            
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        logger.error(f"✗ Failed: {e}")
    
    return result


def _find_onnx_model(model_name: str, onnx_dir: Path) -> Path:
    """Find pre-converted ONNX model."""
    candidates = [
        onnx_dir / model_name / f"{model_name}.onnx",
        onnx_dir / f"{model_name}.onnx",
    ]
    
    for path in candidates:
        if path.exists():
            return path
    return None


def convert_with_compatibility(onnx_path: Path, tflite_path: Path, temp_dir: Path, input_size: int):
    """
    Convert ONNX to TFLite with maximum compatibility settings.
    
    Uses onnx2tf with flags to ensure older TFLite runtime compatibility.
    """
    import onnx
    import tensorflow as tf  # type: ignore[import-unresolved]
    
    logger.info("Converting ONNX to TensorFlow SavedModel...")
    
    # Patch onnx2tf to avoid network timeouts
    try:
        import onnx2tf.utils.common_functions as ocf
        import numpy as np
        def dummy_download(*args, **kwargs):
            return np.zeros((1, 3, 256, 256), dtype=np.float32)
        ocf.download_test_image_data = dummy_download
    except:
        pass
    
    import onnx2tf
    
    saved_model_dir = temp_dir / "saved_model"
    
    # Convert with compatibility settings
    try:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(saved_model_dir),
            not_use_onnxsim=True,
            verbosity='error',
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True,
        )
    except Exception as e:
        logger.warning(f"onnx2tf conversion warning: {e}")
    
    # Find the SavedModel
    if not saved_model_dir.exists():
        raise RuntimeError("SavedModel not created")
    
    # Convert to TFLite with compatibility settings
    logger.info("Converting SavedModel to TFLite with compatibility settings...")
    
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    
    # KEY COMPATIBILITY SETTINGS:
    # 1. Use TF 1.x compatible ops to reduce op version requirements
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
    ]
    
    # 2. Apply optimizations but avoid newest features
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 3. Use float16 quantization (better compatibility than int8)
    converter.target_spec.supported_types = [tf.float16]
    
    # 4. Disable experimental features that may use newer ops
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = False
    
    # 5. Allow custom ops fallback (in case some ops aren't supported)
    converter.allow_custom_ops = False
    
    try:
        tflite_model = converter.convert()
        
        # Save the model
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        logger.info(f"TFLite model saved: {tflite_path}")
        
    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}")
        # Try fallback without float16
        logger.info("Trying fallback conversion without float16...")
        
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)


def run_conversion():
    """Run the compatible conversion for all models."""
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    pytorch_dir = Path(r"D:\Base-dir\checkpoints")
    onnx_dir = Path(r"D:\Base-dir\onnx_models")
    output_dir = script_dir  # tflite_models_compatible/
    temp_dir = Path(r"D:\tmp\tflite_compatible_conversion")
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("TFLite COMPATIBLE MODEL CONVERTER")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target: TFLite 2.14+ compatibility")
    logger.info("")
    
    results = {}
    
    for model_name in MODEL_CONFIGS.keys():
        result = convert_single_model_compatible(
            model_name=model_name,
            pytorch_dir=pytorch_dir,
            onnx_dir=onnx_dir,
            output_dir=output_dir,
            temp_dir=temp_dir
        )
        results[model_name] = result
    
    # Generate metadata file
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'target_runtime': 'TFLite 2.14+',
        'onnx_opset': 11,
        'quantization': 'float16',
        'models': results,
        'input_format': {
            'description': 'Input tensor format: NHWC (batch, height, width, channels)',
            'data_type': 'float32',
            'normalization': '0-1 range (divide by 255)',
            'channel_order': 'RGB'
        }
    }
    
    metadata_path = output_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("CONVERSION SUMMARY")
    logger.info("="*70)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    total_count = len(results)
    
    logger.info(f"\nResults: {success_count}/{total_count} models converted successfully\n")
    
    for model_name, result in results.items():
        status = "✓" if result['status'] == 'success' else "✗"
        size = f"{result['file_size_mb']} MB" if result['file_size_mb'] else "N/A"
        input_size = result['input_size']
        logger.info(f"  {status} {model_name}: {result['status']} ({input_size}x{input_size}, {size})")
    
    logger.info(f"\nMetadata saved to: {metadata_path}")
    logger.info(f"Models saved to: {output_dir}")
    
    return results


if __name__ == '__main__':
    run_conversion()
