"""
ONNX to TFLite via Keras
Uses onnx2keras for conversion.
"""

import os
import json
from pathlib import Path
import tensorflow as tf
import onnx
from onnx2keras import onnx_to_keras

def convert_onnx_to_tflite(onnx_path, output_dir):
    """Convert a single ONNX model to TFLite."""
    model_name = onnx_path.parent.name
    print(f"\n[{model_name}]")
    
    try:
        # Load ONNX
        print("  Loading ONNX...")
        onnx_model = onnx.load(str(onnx_path))
        input_name = onnx_model.graph.input[0].name
        
        # Convert to Keras
        print("  Converting to Keras...")
        keras_model = onnx_to_keras(onnx_model, [input_name], name_policy='renumerate')
        
        # Convert to TFLite
        print("  Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        tflite_model = converter.convert()
        
        # Save
        tflite_path = output_dir / model_name / f"{model_name}.tflite"
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"  SUCCESS - {size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  FAILED - {str(e)}")
        return False


def main():
    onnx_dir = Path(r"D:\Base-dir\onnx_models")
    tflite_dir = Path(r"D:\Base-dir\tflite_models")
    
    print("="*80)
    print("ONNX to TFLite Converter")
    print("="*80)
    
    # Find all ONNX models
    models = []
    for model_dir in onnx_dir.iterdir():
        if model_dir.is_dir():
            onnx_file = model_dir / f"{model_dir.name}.onnx"
            if onnx_file.exists():
                models.append(onnx_file)
    
    print(f"\nFound {len(models)} models\n")
    
    success = 0
    for i, onnx_path in enumerate(models, 1):
        print(f"[{i}/{len(models)}]", end=" ")
        if convert_onnx_to_tflite(onnx_path, tflite_dir):
            success += 1
    
    print("\n" + "="*80)
    print(f"Converted: {success}/{len(models)}")
    print("="*80)


if __name__ == "__main__":
    main()
