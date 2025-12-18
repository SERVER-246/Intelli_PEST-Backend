#!/usr/bin/env python3
"""
ONNX Model Verification and Copy Script
========================================
Verifies all 11 original ONNX models and copies them to the compatible folder.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np

# Class labels for Intelli-PEST (11 classes)
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

NUM_CLASSES = 11

# Source models - mapping from folder name to actual ONNX file name
SOURCE_MODELS = {
    'alexnet': ('D:\\pest-detection-models-v1.0.0\\models\\alexnet', 'alexnet.onnx'),
    'darknet53': ('D:\\pest-detection-models-v1.0.0\\models\\darknet53', 'darknet53.onnx'),
    'efficientnet_b0': ('D:\\pest-detection-models-v1.0.0\\models\\efficientnet_b0', 'efficentnet_b0.onnx'),  # Note: typo in original
    'ensemble_attention': ('D:\\pest-detection-models-v1.0.0\\models\\ensemble_attention', 'attention_fusion.onnx'),
    'ensemble_concat': ('D:\\pest-detection-models-v1.0.0\\models\\ensemble_concat', 'concatination_fusion.onnx'),
    'ensemble_cross': ('D:\\pest-detection-models-v1.0.0\\models\\ensemble_cross', 'cross_attention_fusion.onnx'),
    'inception_v3': ('D:\\pest-detection-models-v1.0.0\\models\\inception_v3', 'inception_v3.onnx'),
    'mobilenet_v2': ('D:\\pest-detection-models-v1.0.0\\models\\mobilenet_v2', 'mobilenet_v2.onnx'),
    'resnet50': ('D:\\pest-detection-models-v1.0.0\\models\\resnet50', 'resnet50.onnx'),
    'super_ensemble': ('D:\\pest-detection-models-v1.0.0\\models\\super_ensemble', 'super_ensemble.onnx'),
    'yolo11n-cls': ('D:\\pest-detection-models-v1.0.0\\models\\yolo11n-cls', 'yolo_11n.onnx'),
}

# Output directory
OUTPUT_DIR = Path(r"D:\Intelli_PEST-Backend\tflite_models_compatible\onnx_models")


def verify_onnx_model(model_path: Path) -> dict:
    """Verify an ONNX model."""
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check model
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        
        # Get input/output info
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        input_shape = list(input_info.shape)
        output_shape = list(output_info.shape)
        
        # Handle dynamic dimensions
        test_input_shape = input_shape.copy()
        if isinstance(test_input_shape[0], str) or test_input_shape[0] is None:
            test_input_shape[0] = 1
        
        # Check number of output classes
        num_output_classes = output_shape[-1] if output_shape else None
        class_mapping_valid = num_output_classes == NUM_CLASSES
        
        # Test inference
        test_input = np.random.randn(*test_input_shape).astype(np.float32)
        output = session.run(None, {input_info.name: test_input})[0]
        
        inference_valid = output.shape[-1] == NUM_CLASSES
        
        # Get opset version
        opset_version = model.opset_import[0].version if model.opset_import else None
        
        return {
            'status': 'success',
            'input_name': input_info.name,
            'input_shape': input_shape,
            'input_dtype': input_info.type,
            'output_name': output_info.name,
            'output_shape': output_shape,
            'num_classes': num_output_classes,
            'class_mapping_valid': class_mapping_valid,
            'inference_valid': inference_valid,
            'opset_version': opset_version,
            'file_size_mb': round(model_path.stat().st_size / 1024 / 1024, 2),
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
        }


def main():
    print("=" * 70)
    print("ONNX MODEL VERIFICATION AND COPY")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'source': 'D:\\pest-detection-models-v1.0.0\\models',
        'destination': str(OUTPUT_DIR),
        'class_names': CLASS_NAMES,
        'expected_num_classes': NUM_CLASSES,
        'models': {},
    }
    
    valid_count = 0
    copied_count = 0
    
    for model_name, (source_dir, onnx_filename) in SOURCE_MODELS.items():
        print(f"\n[{model_name}]")
        
        source_path = Path(source_dir) / onnx_filename
        
        if not source_path.exists():
            print(f"  ✗ Source not found: {source_path}")
            results['models'][model_name] = {
                'status': 'error',
                'error': f'Source not found: {source_path}'
            }
            continue
        
        print(f"  Source: {onnx_filename}")
        
        # Verify the model
        result = verify_onnx_model(source_path)
        results['models'][model_name] = result
        
        if result['status'] == 'success':
            status_icon = "✓" if result['class_mapping_valid'] else "⚠"
            print(f"  {status_icon} Input: {result['input_shape']}")
            print(f"  {status_icon} Output: {result['output_shape']} ({result['num_classes']} classes)")
            print(f"  {status_icon} Size: {result['file_size_mb']} MB")
            print(f"  {status_icon} Opset: {result['opset_version']}")
            print(f"  {status_icon} Inference: {'OK' if result['inference_valid'] else 'FAILED'}")
            
            if result['class_mapping_valid'] and result['inference_valid']:
                valid_count += 1
                
                # Copy to output directory with standardized name
                dest_path = OUTPUT_DIR / f"{model_name}.onnx"
                shutil.copy2(source_path, dest_path)
                print(f"  ✓ Copied to: {dest_path.name}")
                copied_count += 1
                
                result['copied_to'] = str(dest_path)
                
                # Create metadata file
                metadata = {
                    'model_name': model_name,
                    'original_filename': onnx_filename,
                    'input_shape': result['input_shape'],
                    'output_shape': result['output_shape'],
                    'num_classes': result['num_classes'],
                    'class_names': CLASS_NAMES,
                    'opset_version': result['opset_version'],
                    'file_size_mb': result['file_size_mb'],
                    'normalization': {
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225],
                    },
                    'verified_at': datetime.now().isoformat(),
                }
                
                metadata_path = OUTPUT_DIR / f"{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            else:
                print(f"  ⚠ Model has issues, not copied")
        else:
            print(f"  ✗ Error: {result['error']}")
    
    # Create labels file
    labels_path = OUTPUT_DIR / "labels.txt"
    with open(labels_path, 'w') as f:
        for idx in sorted(CLASS_NAMES.keys()):
            f.write(f"{CLASS_NAMES[idx]}\n")
    print(f"\n✓ Created labels.txt")
    
    # Save verification report
    results['summary'] = {
        'total': len(SOURCE_MODELS),
        'valid': valid_count,
        'copied': copied_count,
    }
    
    report_path = OUTPUT_DIR / "verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Models: {len(SOURCE_MODELS)}")
    print(f"Valid Models: {valid_count}")
    print(f"Copied Models: {copied_count}")
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"Report: {report_path}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
