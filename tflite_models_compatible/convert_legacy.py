#!/usr/bin/env python3
"""
TFLite Legacy Compatible Converter
===================================
Alternative converter that uses maximum backward compatibility settings.

This script specifically targets TFLite 2.14.0 compatibility by:
1. Using the oldest possible operation versions
2. Avoiding newer TensorFlow features
3. Using TF 1.x compatible ops where possible

Use this if convert_compatible.py produces models that still don't work.
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('LegacyConverter')

# Model input dimensions - VERIFIED CORRECT VALUES
MODEL_INPUT_SIZES = {
    'mobilenet_v2': 224,
    'resnet50': 224,
    'inception_v3': 299,
    'efficientnet_b0': 224,  # Standard EfficientNet-B0 uses 224, NOT 256
    'darknet53': 224,
    'alexnet': 224,
    'yolo11n-cls': 224,
    'ensemble_attention': 224,
    'ensemble_concat': 224,
    'ensemble_cross': 224,
    'super_ensemble': 224,
}


def export_pytorch_to_onnx_legacy(pt_path: Path, onnx_path: Path, input_size: int):
    """
    Export PyTorch model to ONNX with opset 11 for maximum compatibility.
    
    Using opset 11 produces operators that are more widely supported
    by older ONNX runtimes and TFLite converters.
    """
    import torch
    
    logger.info(f"  Exporting PyTorch -> ONNX (opset 11)")
    
    # Load model
    model = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    # Handle different model save formats
    if isinstance(model, dict):
        if 'model' in model:
            model = model['model']
        elif 'state_dict' in model:
            raise ValueError("Model saved as state_dict only, need full model")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Export with opset 11 - older, more compatible
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,  # Use older opset for compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify and simplify
    import onnx
    import onnxsim  # type: ignore[import-unresolved]
    
    model_onnx = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_onnx)
    
    # Simplify
    model_simplified, check = onnxsim.simplify(model_onnx)
    if check:
        onnx.save(model_simplified, str(onnx_path))
        logger.info(f"  ✓ ONNX exported and simplified: {onnx_path.name}")
    else:
        logger.warning(f"  ! ONNX simplification failed, using original")
    
    return True


def convert_onnx_to_tflite_legacy(onnx_path: Path, tflite_path: Path, input_size: int):
    """
    Convert ONNX to TFLite with MAXIMUM backward compatibility.
    
    Key settings for TFLite 2.14 compatibility:
    - No experimental converters
    - No new quantizers
    - Standard TFLite builtins only
    - No SELECT_TF_OPS (requires flex delegate)
    """
    import tensorflow as tf  # type: ignore[import-unresolved]
    import onnx
    
    # Patch onnx2tf network calls
    try:
        import onnx2tf.utils.common_functions as ocf
        ocf.download_test_image_data = lambda *a, **k: np.zeros((1,3,256,256), dtype=np.float32)
    except:
        pass
    
    import onnx2tf
    
    temp_dir = tflite_path.parent / f"_temp_{tflite_path.stem}"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        logger.info(f"  Step 1: ONNX -> TensorFlow SavedModel")
        
        # Convert ONNX to SavedModel
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(temp_dir),
            not_use_onnxsim=True,
            non_verbose=True,
            verbosity='error',
        )
        
        logger.info(f"  Step 2: SavedModel -> TFLite (Legacy Mode)")
        
        # Convert to TFLite with LEGACY SETTINGS
        converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_dir))
        
        # LEGACY COMPATIBILITY SETTINGS:
        # Only use standard TFLite builtins - no flex ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        
        # DO NOT use experimental features
        converter.experimental_new_converter = False
        converter.experimental_new_quantizer = False
        
        # Use DEFAULT optimization (dynamic range quantization)
        # This should produce FULLY_CONNECTED version that's compatible
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # DO NOT allow custom ops
        converter.allow_custom_ops = False
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"  ✓ Saved: {tflite_path.name} ({len(tflite_model) / 1024 / 1024:.2f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return False
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def convert_all_models():
    """Convert all models with legacy compatibility."""
    
    script_dir = Path(__file__).parent
    pytorch_dir = Path(r"D:\Base-dir\deployment_models")  # PyTorch models
    onnx_dir = Path(r"D:\Base-dir\onnx_models")
    output_dir = script_dir
    temp_dir = Path(r"D:\tmp\tflite_compat_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("TFLITE LEGACY COMPATIBLE CONVERTER")
    logger.info("Target: TFLite Runtime 2.14.0+")
    logger.info("="*60)
    
    results = {}
    
    for model_name, input_size in MODEL_INPUT_SIZES.items():
        logger.info(f"\n[{model_name}] Input: {input_size}x{input_size}")
        
        # First try to find PyTorch model
        pt_candidates = [
            pytorch_dir / f"{model_name}.pt",
            pytorch_dir / f"{model_name}_best.pt",
        ]
        
        pt_path = None
        for candidate in pt_candidates:
            if candidate.exists():
                pt_path = candidate
                break
        
        # Find ONNX model as fallback
        onnx_candidates = [
            onnx_dir / model_name / f"{model_name}.onnx",
            onnx_dir / model_name / f"{model_name}_fixed.onnx",
            onnx_dir / f"{model_name}.onnx",
        ]
        
        onnx_path = None
        for candidate in onnx_candidates:
            if candidate.exists():
                onnx_path = candidate
                break
        
        if not pt_path and not onnx_path:
            logger.warning(f"  ✗ No PyTorch or ONNX model found, skipping")
            results[model_name] = {
                'status': 'skipped',
                'error': 'Model not found',
                'input_size': input_size,
            }
            continue
        
        tflite_path = output_dir / f"{model_name}.tflite"
        model_temp = temp_dir / model_name
        model_temp.mkdir(parents=True, exist_ok=True)
        
        # If we have PyTorch, export to ONNX first with opset 11
        if pt_path and not onnx_path:
            logger.info(f"  Found PyTorch: {pt_path.name}")
            onnx_path = model_temp / f"{model_name}.onnx"
            try:
                export_pytorch_to_onnx_legacy(pt_path, onnx_path, input_size)
            except Exception as e:
                logger.error(f"  ✗ PyTorch->ONNX failed: {e}")
                results[model_name] = {
                    'status': 'failed',
                    'error': f'ONNX export failed: {e}',
                    'input_size': input_size,
                }
                continue
        else:
            logger.info(f"  Found ONNX: {onnx_path.name}")
        
        success = convert_onnx_to_tflite_legacy(onnx_path, tflite_path, input_size)
        
        if success and tflite_path.exists():
            results[model_name] = {
                'status': 'success',
                'input_size': input_size,
                'file_size_mb': round(tflite_path.stat().st_size / 1024 / 1024, 2),
                'tflite_path': str(tflite_path),
            }
        else:
            results[model_name] = {
                'status': 'failed',
                'input_size': input_size,
                'error': 'Conversion failed',
            }
    
    # Save metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'converter': 'legacy_compatible',
        'target_runtime': 'TFLite 2.14.0+',
        'compatibility_mode': 'maximum_backward_compat',
        'settings': {
            'experimental_new_converter': False,
            'experimental_new_quantizer': False,
            'supported_ops': ['TFLITE_BUILTINS'],
            'optimizations': ['DEFAULT'],
        },
        'models': results,
        'input_format': {
            'tensor_format': 'NHWC',
            'data_type': 'float32',
            'normalization': 'divide by 255.0 (0-1 range)',
            'channel_order': 'RGB',
        }
    }
    
    metadata_path = output_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    success = sum(1 for r in results.values() if r.get('status') == 'success')
    total = len(results)
    
    logger.info(f"Converted: {success}/{total}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Metadata: {metadata_path}")
    
    # Print input sizes for Android integration
    logger.info("\n" + "="*60)
    logger.info("ANDROID INPUT SIZE REFERENCE")
    logger.info("="*60)
    for model, info in results.items():
        if info.get('status') == 'success':
            logger.info(f"  {model}: {info['input_size']}x{info['input_size']}")
    
    return results


if __name__ == '__main__':
    convert_all_models()
