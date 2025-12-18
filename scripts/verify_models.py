#!/usr/bin/env python3
"""
Model Verification Script
=========================
Verifies all TFLite and ONNX models for:
1. Correct class mapping (11 classes)
2. Model loads successfully
3. Inference produces valid output
4. Input/output shapes are correct
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# Paths
TFLITE_DIR = Path(r"D:\Intelli_PEST-Backend\tflite_models")
ONNX_DIR = Path(r"D:\Base-dir\onnx_models")

def verify_tflite_model(model_path: Path) -> dict:
    """Verify a TFLite model."""
    try:
        import tensorflow as tf
        
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = input_details[0]['shape'].tolist()
        output_shape = output_details[0]['shape'].tolist()
        
        # Verify output has 11 classes
        num_output_classes = output_shape[-1]
        class_mapping_valid = num_output_classes == NUM_CLASSES
        
        # Test inference
        test_input = np.random.randn(*input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        inference_valid = output.shape[-1] == NUM_CLASSES
        
        return {
            'status': 'success',
            'input_shape': input_shape,
            'output_shape': output_shape,
            'num_classes': num_output_classes,
            'class_mapping_valid': class_mapping_valid,
            'inference_valid': inference_valid,
            'file_size_mb': round(model_path.stat().st_size / 1024 / 1024, 2),
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
        }


def verify_onnx_model(model_path: Path) -> dict:
    """Verify an ONNX model."""
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check model
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        
        # Get input/output info
        session = ort.InferenceSession(str(model_path))
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        input_shape = list(input_info.shape)
        output_shape = list(output_info.shape)
        
        # Handle dynamic dimensions
        if isinstance(input_shape[0], str):
            input_shape[0] = 1
        
        num_output_classes = output_shape[-1] if output_shape else None
        class_mapping_valid = num_output_classes == NUM_CLASSES
        
        # Test inference
        test_input = np.random.randn(*input_shape).astype(np.float32)
        output = session.run(None, {input_info.name: test_input})[0]
        
        inference_valid = output.shape[-1] == NUM_CLASSES
        
        return {
            'status': 'success',
            'input_shape': input_shape,
            'output_shape': output_shape,
            'num_classes': num_output_classes,
            'class_mapping_valid': class_mapping_valid,
            'inference_valid': inference_valid,
            'file_size_mb': round(model_path.stat().st_size / 1024 / 1024, 2),
            'opset_version': model.opset_import[0].version,
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
        }


def main():
    print("=" * 70)
    print("MODEL VERIFICATION REPORT")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'class_names': CLASS_NAMES,
        'expected_num_classes': NUM_CLASSES,
        'tflite_models': {},
        'onnx_models': {},
    }
    
    # Verify TFLite models
    print("\n" + "=" * 50)
    print("TFLITE MODELS")
    print("=" * 50)
    
    tflite_files = list(TFLITE_DIR.rglob("*.tflite"))
    tflite_valid = 0
    
    for tflite_path in sorted(tflite_files):
        model_name = tflite_path.stem
        print(f"\n[{model_name}]")
        
        result = verify_tflite_model(tflite_path)
        results['tflite_models'][model_name] = result
        
        if result['status'] == 'success':
            status_icon = "✓" if result['class_mapping_valid'] else "⚠"
            print(f"  {status_icon} Input: {result['input_shape']}")
            print(f"  {status_icon} Output: {result['output_shape']} ({result['num_classes']} classes)")
            print(f"  {status_icon} Size: {result['file_size_mb']} MB")
            print(f"  {status_icon} Inference: {'OK' if result['inference_valid'] else 'FAILED'}")
            
            if result['class_mapping_valid'] and result['inference_valid']:
                tflite_valid += 1
        else:
            print(f"  ✗ Error: {result['error']}")
    
    # Verify ONNX models
    print("\n" + "=" * 50)
    print("ONNX MODELS")
    print("=" * 50)
    
    onnx_files = list(ONNX_DIR.rglob("*.onnx"))
    onnx_valid = 0
    
    if not onnx_files:
        print("\n  ⚠ No ONNX files found in", ONNX_DIR)
    
    for onnx_path in sorted(onnx_files):
        model_name = onnx_path.stem
        print(f"\n[{model_name}]")
        
        result = verify_onnx_model(onnx_path)
        results['onnx_models'][model_name] = result
        
        if result['status'] == 'success':
            status_icon = "✓" if result['class_mapping_valid'] else "⚠"
            print(f"  {status_icon} Input: {result['input_shape']}")
            print(f"  {status_icon} Output: {result['output_shape']} ({result['num_classes']} classes)")
            print(f"  {status_icon} Size: {result['file_size_mb']} MB")
            print(f"  {status_icon} Opset: {result['opset_version']}")
            print(f"  {status_icon} Inference: {'OK' if result['inference_valid'] else 'FAILED'}")
            
            if result['class_mapping_valid'] and result['inference_valid']:
                onnx_valid += 1
        else:
            print(f"  ✗ Error: {result['error']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"TFLite Models: {tflite_valid}/{len(tflite_files)} valid")
    print(f"ONNX Models: {onnx_valid}/{len(onnx_files)} valid")
    
    results['summary'] = {
        'tflite_total': len(tflite_files),
        'tflite_valid': tflite_valid,
        'onnx_total': len(onnx_files),
        'onnx_valid': onnx_valid,
    }
    
    # Save report
    report_path = Path(__file__).parent / "verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    main()
