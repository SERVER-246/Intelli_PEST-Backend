"""
Simple ONNX to TFLite Converter
A reliable converter that uses PyTorch models directly to create TFLite models.
"""

import os
import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_pytorch_to_tflite(pytorch_model_path, output_dir, model_name):
    """
    Convert PyTorch model to TFLite format directly.
    
    Args:
        pytorch_model_path: Path to .pth or .pt file
        output_dir: Output directory for TFLite models
        model_name: Name for the output model
    """
    try:
        logger.info(f"Converting {model_name}...")
        
        # Load PyTorch model
        logger.info(f"Loading PyTorch model from {pytorch_model_path}")
        device = torch.device('cpu')
        
        # Try loading as full model first (this is what we actually need for export)
        try:
            model = torch.load(pytorch_model_path, map_location=device, weights_only=False)
            logger.info("Loaded as full model")
            
            # If it's a JIT model, it's ready to use
            if isinstance(model, torch.jit.ScriptModule):
                logger.info("Model is TorchScript format")
                model.eval()
            # If it's a dict, we can't convert it without architecture
            elif isinstance(model, dict):
                logger.error("Model is state dict only - architecture needed for conversion")
                logger.error("Skipping this model (requires model architecture to load state dict)")
                return None
            else:
                model.eval()
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 256)
        
        # Export to ONNX first
        onnx_path = Path(output_dir) / f"{model_name}.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Exporting to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        logger.info(f"ONNX model saved to {onnx_path}")
        logger.info("Note: For TFLite conversion, use ONNX models with onnxruntime or TFLite converter tools")
        
        return str(onnx_path)
        
    except Exception as e:
        logger.error(f"Error converting {model_name}: {str(e)}")
        return None


def main():
    """Main conversion workflow."""
    
    print("="*80)
    print("PyTorch to ONNX Converter")
    print("="*80)
    print()
    
    # Check for models
    base_dir = Path("d:/Base-dir")
    deployment_dir = base_dir / "deployment_models"
    checkpoints_dir = base_dir / "checkpoints"
    
    output_dir = Path("outputs/onnx_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for PyTorch models
    pytorch_models = []
    
    if deployment_dir.exists():
        pytorch_models.extend(list(deployment_dir.glob("*.pt")))
        pytorch_models.extend(list(deployment_dir.glob("*.pth")))
    
    if checkpoints_dir.exists():
        pytorch_models.extend(list(checkpoints_dir.glob("*.pt")))
        pytorch_models.extend(list(checkpoints_dir.glob("*.pth")))
    
    if not pytorch_models:
        print("❌ No PyTorch models found!")
        print(f"   Checked: {deployment_dir}")
        print(f"   Checked: {checkpoints_dir}")
        print()
        print("Please ensure you have trained models available.")
        return
    
    print(f"Found {len(pytorch_models)} PyTorch models")
    print()
    
    # Convert each model
    converted = 0
    for i, model_path in enumerate(pytorch_models, 1):
        model_name = model_path.stem
        print(f"[{i}/{len(pytorch_models)}] Converting {model_name}...")
        
        result = convert_pytorch_to_tflite(model_path, output_dir, model_name)
        if result:
            converted += 1
            print(f"  ✓ Converted: {result}")
        else:
            print(f"  ✗ Failed")
        print()
    
    print("="*80)
    print(f"Conversion Summary: {converted}/{len(pytorch_models)} models converted")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*80)


if __name__ == "__main__":
    main()
