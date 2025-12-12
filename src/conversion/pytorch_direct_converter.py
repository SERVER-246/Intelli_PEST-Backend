"""
Direct PyTorch to TFLite Converter
===================================
Uses PyTorch Mobile export path to avoid ONNX conversion issues.
"""

import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
import json
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PyTorchToTFLite')


def convert_pytorch_to_tflite_via_torchscript(
    pytorch_path: Path,
    tflite_path: Path,
    input_shape=(1, 3, 256, 256)
) -> dict:
    """
    Convert PyTorch model to TFLite using TorchScript + ONNX + TFLite.
    This bypasses onnx2keras issues by using simpler ONNX opset.
    """
    logger.info(f"Converting {pytorch_path.name}...")
    
    result = {'status': 'pending', 'pytorch_path': str(pytorch_path)}
    
    try:
        # Load PyTorch model
        logger.info("  Loading PyTorch model...")
        model = torch.jit.load(str(pytorch_path), map_location='cpu')
        model.eval()
        
        # Create example input
        example_input = torch.randn(*input_shape)
        
        # Trace the model
        logger.info("  Tracing model...")
        traced_model = torch.jit.trace(model, example_input)
        
        # Export to ONNX with older opset for better compatibility
        temp_onnx = tflite_path.parent / f"temp_{tflite_path.stem}.onnx"
        logger.info("  Exporting to ONNX (opset 9 for compatibility)...")
        
        torch.onnx.export(
            traced_model,
            example_input,
            str(temp_onnx),
            opset_version=9,  # Use older opset that's more compatible
            input_names=['input'],
            output_names=['output'],
            do_constant_folding=True
        )
        
        # Load ONNX and convert via TF
        logger.info("  Converting ONNX to TFLite...")
        
        # Use TF's ONNX import if available
        try:
            import onnx
            import onnx_tf
            from onnx_tf.backend import prepare
            
            onnx_model = onnx.load(str(temp_onnx))
            tf_rep = prepare(onnx_model)
            
            # Export to SavedModel
            temp_saved_model = tflite_path.parent / f"temp_{tflite_path.stem}_saved_model"
            tf_rep.export_graph(str(temp_saved_model))
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_saved_model))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
            tflite_model = converter.convert()
            
            # Cleanup
            import shutil
            if temp_saved_model.exists():
                shutil.rmtree(temp_saved_model)
            
        except Exception as e:
            logger.warning(f"onnx-tf path failed: {e}")
            # Fallback: Try direct TFLite from ONNX using experimental converter
            raise NotImplementedError("Direct ONNX->TFLite not yet implemented")
        
        # Save TFLite
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        result['tflite_path'] = str(tflite_path)
        result['size_mb'] = round(size_mb, 2)
        result['status'] = 'success'
        
        logger.info(f"  ✓ SUCCESS: {size_mb:.2f} MB")
        
        # Cleanup temp ONNX
        if temp_onnx.exists():
            temp_onnx.unlink()
        
        return result
        
    except Exception as e:
        logger.error(f"  ✗ FAILED: {str(e)}")
        result['status'] = 'failed'
        result['error'] = str(e)
        return result


def main():
    """Test on remaining models."""
    deployment_dir = Path(r"D:\Base-dir\deployment_models")
    tflite_dir = Path(r"D:\Base-dir\tflite_models")
    
    remaining_models = [
        'mobilenet_v2',
        'darknet53',
        'ensemble_attention',
        'ensemble_concat',
        'ensemble_cross',
        'super_ensemble'
    ]
    
    print("="*70)
    print("PyTorch to TFLite Direct Converter")
    print("="*70)
    print(f"\nConverting {len(remaining_models)} models\n")
    
    results = {}
    for model_name in remaining_models:
        print(f"\n[{model_name}]")
        
        # Find PyTorch model
        pt_path = deployment_dir / f"{model_name}.pt"
        if not pt_path.exists():
            # Try deployment subdirectory
            subdir = deployment_dir / f"{model_name}_deployment" / f"{model_name}.pt"
            if subdir.exists():
                pt_path = subdir
            else:
                print(f"  ✗ PyTorch model not found")
                results[model_name] = {'status': 'skipped', 'reason': 'pt file not found'}
                continue
        
        output_dir = tflite_dir / model_name
        tflite_path = output_dir / f"{model_name}.tflite"
        
        result = convert_pytorch_to_tflite_via_torchscript(pt_path, tflite_path)
        results[model_name] = result
    
    # Save report
    report_path = tflite_dir / 'pytorch_direct_conversion_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    successful = sum(1 for r in results.values() if r.get('status') == 'success')
    print(f"\n{'='*70}")
    print(f"Converted: {successful}/{len(remaining_models)}")
    print(f"Report: {report_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
