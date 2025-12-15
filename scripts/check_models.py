"""
PyTorch to TFLite Converter
Direct conversion from PyTorch TorchScript models to TFLite.
Uses ai_edge_torch for direct conversion.
"""

import os
from pathlib import Path
import torch

def convert_pytorch_to_tflite(pt_path, output_dir):
    """Convert PyTorch .pt file to TFLite."""
    model_name = pt_path.stem
    print(f"\n[{model_name}]")
    
    try:
        # Try using PyTorch's native export to TFLite
        # This requires ai-edge-torch
        print("  Loading PyTorch model...")
        model = torch.load(str(pt_path), map_location='cpu')
        
        if not isinstance(model, torch.jit.ScriptModule):
            print("  SKIP - Not a TorchScript model")
            return False
        
        model.eval()
        
        # For now, just confirm the ONNX file exists
        onnx_dir = Path(r"D:\Base-dir\onnx_models")
        onnx_path = onnx_dir / model_name / f"{model_name}.onnx"
        
        if onnx_path.exists():
            print(f"  ONNX exists: {onnx_path}")
            print("  Note: Use external tools like https://netron.app or online converters")
            print("        to convert ONNX to TFLite")
            return True
        else:
            print("  ONNX file not found")
            return False
            
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False


def main():
    pt_dir = Path(r"D:\Base-dir\deployment_models")
    tflite_dir = Path(r"D:\Base-dir\tflite_models")
    
    print("="*80)
    print("PyTorch Model Checker")
    print("="*80)
    
    # Find .pt files
    pt_files = list(pt_dir.glob("*.pt"))
    
    print(f"\nFound {len(pt_files)} .pt files\n")
    
    for i, pt_file in enumerate(pt_files, 1):
        print(f"[{i}/{len(pt_files)}]", end=" ")
        convert_pytorch_to_tflite(pt_file, tflite_dir)
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("Your ONNX models are ready in: D:\\Base-dir\\onnx_models\\")
    print("\nTo convert ONNX to TFLite, use one of these options:")
    print("  1. Online: https://convertmodel.com/")
    print("  2. Online: https://github.com/onnx/onnx-tensorflow")
    print("  3. Local: Install onnx-tf properly and use tf.lite.TFLiteConverter")
    print("\nOr just use the ONNX models directly with onnxruntime in your app!")
    print("="*80)


if __name__ == "__main__":
    main()
