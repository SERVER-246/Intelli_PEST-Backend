#!/usr/bin/env python3
"""
Android-Compatible TFLite Model Converter
==========================================
Converts all PyTorch/ONNX models to TFLite format compatible with
Android TFLite Runtime 2.14.0+.

Features:
- Handles PyTorch (.pt) and ONNX (.onnx) models
- Uses older opset versions for compatibility
- Embeds proper class labels metadata
- Creates organized output structure for Android deployment
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AndroidCompatConverter')

# Class labels for Intelli-PEST
CLASS_NAMES = {
    0: "Healthy",
    1: "Internode borer",
    2: "Pink borer",
    3: "Rat damage",
    4: "Stalk borer",
    5: "Top borer",
    6: "Army worm",
    7: "Mealy bug",
    8: "Porcupine damage",
    9: "Root borer",
    10: "Termite"
}

# Model configurations
MODEL_CONFIGS = {
    'mobilenet_v2': {'input_size': 224, 'type': 'base'},
    'resnet50': {'input_size': 224, 'type': 'base'},
    'inception_v3': {'input_size': 299, 'type': 'base'},
    'efficientnet_b0': {'input_size': 224, 'type': 'base'},
    'darknet53': {'input_size': 224, 'type': 'base'},
    'alexnet': {'input_size': 224, 'type': 'base'},
    'yolo11n-cls': {'input_size': 224, 'type': 'base'},
    'ensemble_attention': {'input_size': 224, 'type': 'ensemble'},
    'ensemble_concat': {'input_size': 224, 'type': 'ensemble'},
    'ensemble_cross': {'input_size': 224, 'type': 'ensemble'},
    'super_ensemble': {'input_size': 224, 'type': 'ensemble'},
}

# Paths
BASE_DIR = Path(r"D:\Base-dir")
PYTORCH_DIR = BASE_DIR / "deployment_models"
ONNX_DIR = BASE_DIR / "onnx_models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = Path(__file__).parent / "android_models"
TEMP_DIR = Path(r"D:\tmp\android_compat_conversion")


def setup_output_directory():
    """Create clean output directory structure."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create labels file
    labels_file = OUTPUT_DIR / "labels.txt"
    with open(labels_file, 'w') as f:
        for idx in sorted(CLASS_NAMES.keys()):
            f.write(f"{CLASS_NAMES[idx]}\n")
    
    logger.info(f"Created output directory: {OUTPUT_DIR}")
    return OUTPUT_DIR


def find_model_file(model_name: str, config: dict) -> Tuple[Optional[Path], str]:
    """Find the best available model file (PyTorch or ONNX)."""
    
    # Priority 1: ONNX model (already exported)
    onnx_candidates = [
        ONNX_DIR / model_name / f"{model_name}_fixed.onnx",
        ONNX_DIR / model_name / f"{model_name}.onnx",
        ONNX_DIR / f"{model_name}.onnx",
    ]
    
    for candidate in onnx_candidates:
        if candidate.exists():
            return candidate, 'onnx'
    
    # Priority 2: PyTorch model
    pt_candidates = [
        PYTORCH_DIR / f"{model_name}.pt",
        PYTORCH_DIR / f"{model_name}_best.pt",
    ]
    
    for candidate in pt_candidates:
        if candidate.exists():
            return candidate, 'pytorch'
    
    # Priority 3: Ensemble checkpoints
    if config['type'] == 'ensemble':
        checkpoint_candidates = [
            CHECKPOINTS_DIR / f"{model_name}.pth",
            CHECKPOINTS_DIR / f"{model_name}.pt",
        ]
        for candidate in checkpoint_candidates:
            if candidate.exists():
                return candidate, 'checkpoint'
    
    return None, 'not_found'


def export_pytorch_to_onnx(pt_path: Path, onnx_path: Path, input_size: int) -> bool:
    """Export PyTorch model to ONNX with opset 11 for compatibility."""
    try:
        import torch
        import onnx
        
        logger.info(f"    Exporting PyTorch -> ONNX (opset 11)")
        
        # Load model
        model = torch.load(pt_path, map_location='cpu', weights_only=False)
        
        # Handle different model save formats
        if isinstance(model, dict):
            if 'model' in model:
                model = model['model']
            elif 'state_dict' in model:
                logger.error("    Model saved as state_dict only, need architecture")
                return False
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        # Export with opset 11
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        model_onnx = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_onnx)
        
        # Try to simplify
        try:
            import onnxsim  # type: ignore[import-unresolved]
            model_simplified, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simplified, str(onnx_path))
                logger.info(f"    ✓ ONNX simplified successfully")
        except Exception as e:
            logger.warning(f"    ! ONNX simplification skipped: {e}")
        
        logger.info(f"    ✓ ONNX export complete: {onnx_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"    ✗ PyTorch->ONNX failed: {e}")
        return False


def convert_onnx_to_tflite_tf2(onnx_path: Path, tflite_path: Path, input_size: int) -> bool:
    """
    Convert ONNX to TFLite using TensorFlow's native converter.
    This method tends to produce more compatible models.
    """
    try:
        import tensorflow as tf  # type: ignore[import-unresolved]
        import onnx
        from onnx_tf.backend import prepare  # type: ignore[import-unresolved]
        
        logger.info(f"    Converting ONNX -> TFLite (TF native method)")
        
        # Create temp directory for SavedModel
        temp_saved_model = TEMP_DIR / f"{tflite_path.stem}_saved_model"
        if temp_saved_model.exists():
            shutil.rmtree(temp_saved_model)
        
        # Load ONNX model
        onnx_model = onnx.load(str(onnx_path))
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(temp_saved_model))
        
        logger.info(f"    ✓ Converted to SavedModel")
        
        # Convert SavedModel to TFLite with compatibility settings
        converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_saved_model))
        
        # Compatibility settings for TFLite 2.14
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        
        # Disable experimental features
        converter.experimental_new_converter = True  # Use MLIR but stable features only
        converter.experimental_new_quantizer = False
        
        # Optional: Dynamic range quantization (smaller, still compatible)
        # Uncomment if you want quantized models
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Cleanup
        if temp_saved_model.exists():
            shutil.rmtree(temp_saved_model, ignore_errors=True)
        
        logger.info(f"    ✓ TFLite saved: {tflite_path.name} ({len(tflite_model)/1024/1024:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"    ✗ TF native conversion failed: {e}")
        return False


def convert_onnx_to_tflite_onnx2tf(onnx_path: Path, tflite_path: Path, input_size: int) -> bool:
    """
    Convert ONNX to TFLite using onnx2tf.
    Alternative method that sometimes handles complex models better.
    """
    try:
        import onnx2tf
        
        logger.info(f"    Converting ONNX -> TFLite (onnx2tf method)")
        
        # Convert with onnx2tf
        temp_output = TEMP_DIR / f"{tflite_path.stem}_onnx2tf"
        if temp_output.exists():
            shutil.rmtree(temp_output)
        
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(temp_output),
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True,
        )
        
        # Find the generated tflite file
        tflite_files = list(temp_output.glob("*.tflite"))
        if not tflite_files:
            # Check for saved_model and convert
            saved_model_path = temp_output / "saved_model"
            if saved_model_path.exists():
                import tensorflow as tf  # type: ignore[import-unresolved]
                converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                tflite_model = converter.convert()
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)
                logger.info(f"    ✓ TFLite saved: {tflite_path.name}")
                return True
            else:
                logger.error(f"    ✗ No TFLite file generated")
                return False
        
        # Copy the best tflite file
        # Prefer float32 for compatibility
        source_tflite = None
        for tf in tflite_files:
            if 'float32' in tf.name.lower():
                source_tflite = tf
                break
        if not source_tflite:
            source_tflite = tflite_files[0]
        
        shutil.copy(source_tflite, tflite_path)
        
        # Cleanup
        if temp_output.exists():
            shutil.rmtree(temp_output, ignore_errors=True)
        
        logger.info(f"    ✓ TFLite saved: {tflite_path.name} ({tflite_path.stat().st_size/1024/1024:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"    ✗ onnx2tf conversion failed: {e}")
        return False


def verify_tflite_compatibility(tflite_path: Path) -> dict:
    """Verify TFLite model can be loaded and check for compatibility issues."""
    try:
        import tensorflow as tf  # type: ignore[import-unresolved]
        
        # Try to load with TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = input_details[0]['shape'].tolist()
        output_shape = output_details[0]['shape'].tolist()
        
        return {
            'valid': True,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'input_dtype': str(input_details[0]['dtype']),
            'output_dtype': str(output_details[0]['dtype']),
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
        }


def create_model_metadata(model_name: str, config: dict, tflite_path: Path, verification: dict) -> dict:
    """Create metadata JSON for the model."""
    metadata = {
        'model_name': model_name,
        'model_type': config['type'],
        'input_size': config['input_size'],
        'num_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES,
        'input_format': {
            'shape': verification.get('input_shape', [1, config['input_size'], config['input_size'], 3]),
            'dtype': verification.get('input_dtype', 'float32'),
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
            }
        },
        'output_format': {
            'shape': verification.get('output_shape', [1, len(CLASS_NAMES)]),
            'dtype': verification.get('output_dtype', 'float32'),
            'type': 'softmax_probabilities',
        },
        'file_size_mb': round(tflite_path.stat().st_size / 1024 / 1024, 2) if tflite_path.exists() else 0,
        'compatibility': {
            'min_tflite_runtime': '2.14.0',
            'android_min_sdk': 21,
        },
        'generated_at': datetime.now().isoformat(),
    }
    
    return metadata


def convert_all_models():
    """Main conversion function."""
    
    logger.info("=" * 70)
    logger.info("ANDROID-COMPATIBLE TFLITE MODEL CONVERTER")
    logger.info("Target: TFLite Runtime 2.14.0+ (Android)")
    logger.info("=" * 70)
    
    # Setup
    setup_output_directory()
    
    results = {}
    successful = 0
    failed = 0
    skipped = 0
    
    for model_name, config in MODEL_CONFIGS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"[{model_name}] Input: {config['input_size']}x{config['input_size']}, Type: {config['type']}")
        logger.info(f"{'='*50}")
        
        # Find source model
        source_path, source_type = find_model_file(model_name, config)
        
        if source_type == 'not_found':
            logger.warning(f"  ✗ No source model found, skipping")
            results[model_name] = {'status': 'skipped', 'reason': 'source not found'}
            skipped += 1
            continue
        
        logger.info(f"  Found {source_type} model: {source_path.name}")
        
        # Prepare paths
        model_temp_dir = TEMP_DIR / model_name
        model_temp_dir.mkdir(parents=True, exist_ok=True)
        
        tflite_path = OUTPUT_DIR / f"{model_name}.tflite"
        
        # Convert to ONNX first if needed
        if source_type in ['pytorch', 'checkpoint']:
            onnx_path = model_temp_dir / f"{model_name}.onnx"
            if not export_pytorch_to_onnx(source_path, onnx_path, config['input_size']):
                logger.error(f"  ✗ Failed to export to ONNX")
                results[model_name] = {'status': 'failed', 'reason': 'ONNX export failed'}
                failed += 1
                continue
        else:
            onnx_path = source_path
        
        # Try conversion methods in order of preference
        conversion_success = False
        
        # Method 1: TF native (usually most compatible)
        logger.info(f"  Attempting Method 1: TensorFlow native converter")
        conversion_success = convert_onnx_to_tflite_tf2(onnx_path, tflite_path, config['input_size'])
        
        # Method 2: onnx2tf (fallback)
        if not conversion_success:
            logger.info(f"  Attempting Method 2: onnx2tf converter")
            conversion_success = convert_onnx_to_tflite_onnx2tf(onnx_path, tflite_path, config['input_size'])
        
        if conversion_success and tflite_path.exists():
            # Verify the model
            verification = verify_tflite_compatibility(tflite_path)
            
            if verification['valid']:
                # Create metadata
                metadata = create_model_metadata(model_name, config, tflite_path, verification)
                metadata_path = OUTPUT_DIR / f"{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                results[model_name] = {
                    'status': 'success',
                    'file': str(tflite_path.name),
                    'size_mb': metadata['file_size_mb'],
                    'input_shape': verification['input_shape'],
                    'output_shape': verification['output_shape'],
                }
                successful += 1
                logger.info(f"  ✓ SUCCESS: {tflite_path.name}")
            else:
                results[model_name] = {
                    'status': 'failed',
                    'reason': f"Verification failed: {verification.get('error', 'unknown')}",
                }
                failed += 1
                logger.error(f"  ✗ Verification failed")
        else:
            results[model_name] = {'status': 'failed', 'reason': 'All conversion methods failed'}
            failed += 1
            logger.error(f"  ✗ All conversion methods failed")
    
    # Create summary report
    summary = {
        'conversion_date': datetime.now().isoformat(),
        'target_runtime': 'TFLite 2.14.0+',
        'total_models': len(MODEL_CONFIGS),
        'successful': successful,
        'failed': failed,
        'skipped': skipped,
        'class_names': CLASS_NAMES,
        'models': results,
    }
    
    report_path = OUTPUT_DIR / "conversion_report.json"
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total Models: {len(MODEL_CONFIGS)}")
    logger.info(f"  Successful:   {successful}")
    logger.info(f"  Failed:       {failed}")
    logger.info(f"  Skipped:      {skipped}")
    logger.info(f"\n  Output:       {OUTPUT_DIR}")
    logger.info(f"  Report:       {report_path}")
    logger.info("=" * 70)
    
    # List successful models
    if successful > 0:
        logger.info("\nSuccessful Models:")
        for name, result in results.items():
            if result['status'] == 'success':
                logger.info(f"  - {name}.tflite ({result['size_mb']} MB)")
    
    return summary


if __name__ == "__main__":
    convert_all_models()
