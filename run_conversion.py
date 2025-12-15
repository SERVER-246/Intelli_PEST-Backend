#!/usr/bin/env python
"""
Master Script: PyTorch to TFLite Conversion Pipeline
====================================================
Complete end-to-end conversion of all pest detection models from PyTorch to TensorFlow Lite.

Usage:
    python run_conversion.py                    # Convert all 11 models
    python run_conversion.py --model mobilenet_v2  # Convert single model
    python run_conversion.py --help             # Show help

Requirements:
    - Python 3.10+
    - Create venv: python -m venv venv_tflite
    - Install: pip install -r requirements_tflite.txt
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from conversion.pytorch_to_tflite_quantized import PyTorchToTFLiteQuantized


def main():
    """Execute the conversion pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert pest detection models to TFLite format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Convert all models
    python run_conversion.py
    
    # Convert single model
    python run_conversion.py --model mobilenet_v2
    
    # Custom directories
    python run_conversion.py --input_dir D:\\models --output_dir ./tflite_output
        '''
    )
    
    parser.add_argument('--model', type=str, help='Specific model to convert')
    parser.add_argument(
        '--input_dir',
        type=str,
        default=r'D:\Base-dir\deployment_models',
        help='Directory containing PyTorch .pt models'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tflite_models',
        help='Directory to save TFLite models'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Create converter
    pytorch_dir = Path(args.input_dir)
    tflite_dir = Path(args.output_dir)
    
    if not pytorch_dir.exists():
        print(f"Error: Input directory not found: {pytorch_dir}")
        return 1
    
    tflite_dir.mkdir(parents=True, exist_ok=True)
    
    converter = PyTorchToTFLiteQuantized(pytorch_dir, tflite_dir)
    
    # List of all available models
    ALL_MODELS = [
        'mobilenet_v2',
        'darknet53',
        'alexnet',
        'resnet50',
        'inception_v3',
        'efficientnet_b0',
        'yolo11n-cls',
        'ensemble_attention',
        'ensemble_concat',
        'ensemble_cross',
        'super_ensemble'
    ]
    
    if args.model:
        # Single model conversion
        print(f"\n{'='*70}")
        print(f"Converting single model: {args.model}")
        print(f"{'='*70}\n")
        result = converter.convert_model(args.model)
        
        if result.get('status') == 'success':
            print(f"\n✓ SUCCESS: {args.model}")
            print(f"  TFLite: {result.get('tflite_size_mb')} MB")
            print(f"  Compression: {result.get('compression_ratio')}%")
            return 0
        else:
            print(f"\n✗ FAILED: {args.model}")
            print(f"  Error: {result.get('error')}")
            return 1
    else:
        # Batch conversion
        print(f"\n{'='*70}")
        print(f"PyTorch to TFLite Conversion Pipeline")
        print(f"Converting {len(ALL_MODELS)} models with Dynamic Range Quantization")
        print(f"{'='*70}\n")
        
        results = converter.convert_all(ALL_MODELS)
        
        # Print summary
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        failed = len(ALL_MODELS) - successful
        
        print(f"\n{'='*70}")
        print(f"CONVERSION COMPLETE")
        print(f"{'='*70}")
        print(f"Total:      {len(ALL_MODELS)}")
        print(f"Successful: {successful}")
        print(f"Failed:     {failed}")
        
        if successful == len(ALL_MODELS):
            print(f"\n✓ ALL MODELS SUCCESSFULLY CONVERTED!")
            return 0
        else:
            print(f"\n⚠ Partial success: {successful}/{len(ALL_MODELS)} converted")
            return 1


if __name__ == '__main__':
    sys.exit(main())
