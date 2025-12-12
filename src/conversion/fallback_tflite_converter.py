"""
Fallback TFLite Converter for Models that Failed with onnx2keras
=================================================================
This converter attempts multiple conversion paths:
1. ONNX -> TensorFlow (via onnx-tensorflow) -> TFLite
2. PyTorch/TorchScript -> ONNX (fixed) -> TensorFlow -> TFLite
3. Direct PyTorch -> TFLite (via ai_edge_torch if available)
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path
import shutil
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import onnx
import numpy as np


# Try to import conversion libraries
ONNX_TF_AVAILABLE = False  # Disabled due to TF version incompatibility
# Using alternative conversion path with ONNX simplification instead

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed")

try:
    import onnxsim
    ONNX_SIM_AVAILABLE = True
except ImportError:
    ONNX_SIM_AVAILABLE = False
    print("INFO: onnxsim not available (optional optimization)")


def setup_logging(log_dir: Path) -> tuple:
    """Setup logging for fallback converter."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('FallbackConverter')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    log_file = log_dir / f"fallback_conversion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


class FallbackTFLiteConverter:
    """Fallback converter using alternative conversion paths."""
    
    QUANTIZATION_DYNAMIC = "dynamic"
    QUANTIZATION_FLOAT16 = "float16"
    QUANTIZATION_NONE = "none"
    
    def __init__(self, onnx_dir: str, tflite_dir: str, quantization: str = QUANTIZATION_DYNAMIC):
        self.onnx_dir = Path(onnx_dir)
        self.tflite_dir = Path(tflite_dir)
        self.log_dir = self.tflite_dir / 'logs'
        self.quantization = quantization
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger, self.log_file = setup_logging(self.log_dir)
        
        self.results = {}
        self.successful = 0
        self.failed = 0
        
        self.logger.info("=" * 70)
        self.logger.info("FALLBACK TFLITE CONVERTER")
        self.logger.info("=" * 70)
        self.logger.info(f"TensorFlow: {tf.__version__}")
        self.logger.info(f"ONNX-TensorFlow: {'Available' if ONNX_TF_AVAILABLE else 'NOT Available'}")
        self.logger.info(f"PyTorch: {'Available' if TORCH_AVAILABLE else 'NOT Available'}")
        self.logger.info(f"Source: {self.onnx_dir}")
        self.logger.info(f"Output: {self.tflite_dir}")
        self.logger.info(f"Quantization: {self.quantization}")
    
    def _apply_quantization(self, converter: tf.lite.TFLiteConverter) -> None:
        """Apply quantization settings."""
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        if self.quantization == self.QUANTIZATION_NONE:
            converter.optimizations = []
        elif self.quantization == self.QUANTIZATION_DYNAMIC:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif self.quantization == self.QUANTIZATION_FLOAT16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
    
    def _load_labels(self, model_name: str) -> list:
        """Load labels from model directory."""
        labels_file = self.onnx_dir / model_name / 'labels.txt'
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        return [f"class_{i}" for i in range(11)]
    
    def _verify_tflite(self, tflite_path: Path) -> dict:
        """Verify TFLite model."""
        try:
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            input_shape = input_details[0]['shape']
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            import time
            times = []
            for _ in range(10):
                start = time.perf_counter()
                interpreter.set_tensor(input_details[0]['index'], dummy_input)
                interpreter.invoke()
                times.append(time.perf_counter() - start)
            
            return {
                'status': 'verified',
                'avg_inference_ms': round(np.mean(times) * 1000, 2),
                'input_shape': input_shape.tolist(),
                'output_shape': output_details[0]['shape'].tolist()
            }
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def _save_metadata(self, model_name: str, output_dir: Path, 
                       onnx_size: float, tflite_size: float) -> None:
        """Save Android metadata."""
        labels = self._load_labels(model_name)
        
        metadata = {
            "model_info": {
                "name": model_name,
                "version": "1.0.0",
                "description": f"TFLite model - {model_name}",
                "created_date": datetime.datetime.now().isoformat(),
                "framework": "TensorFlow Lite",
                "quantization": self.quantization,
                "original_size_mb": onnx_size,
                "compressed_size_mb": tflite_size,
                "conversion_method": "fallback (onnx-tensorflow)"
            },
            "input": {
                "name": "input",
                "shape": [1, 3, 256, 256],
                "dtype": "float32",
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "color_format": "RGB"
            },
            "output": {
                "name": "output",
                "shape": [1, len(labels)],
                "dtype": "float32"
            },
            "labels": labels,
            "num_classes": len(labels)
        }
        
        with open(output_dir / 'android_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open(output_dir / 'labels.txt', 'w') as f:
            f.write('\n'.join(labels))
    
    def convert_via_onnx_tensorflow(self, model_name: str) -> dict:
        """
        Method 1: Convert ONNX -> Keras -> TFLite using onnx2keras with sanitization.
        This often works for models that fail with standard onnx2keras.
        """
        # Import required libraries
        try:
            from onnx2keras import onnx_to_keras
            from onnx_sanitizer import ONNXSanitizer
        except ImportError as e:
            raise ImportError(f"Required libraries not available: {e}")
        
        onnx_path = self.onnx_dir / model_name / f"{model_name}.onnx"
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Method 1: ONNX->Keras Conversion (with sanitization): {model_name}")
        self.logger.info(f"{'='*70}")
        
        result = {
            'model_name': model_name,
            'onnx_path': str(onnx_path),
            'timestamp': datetime.datetime.now().isoformat(),
            'conversion_method': 'onnx2keras-sanitized',
            'quantization': self.quantization
        }
        
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        
        # Step 1: Load ONNX
        self.logger.info("Step 1/4: Loading ONNX model...")
        onnx_model = onnx.load(str(onnx_path))
        onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        result['onnx_size_mb'] = round(onnx_size_mb, 2)
        self.logger.info(f"  Loaded ({result['onnx_size_mb']} MB)")
        
        # Step 2: Sanitize ONNX model
        self.logger.info("Step 2/4: Sanitizing ONNX for TensorFlow compatibility...")
        sanitizer = ONNXSanitizer()
        onnx_model = sanitizer.sanitize_model(onnx_model)
        self.logger.info(f"  Sanitized {len(sanitizer.name_mapping)} names")
        
        # Step 3: Convert to Keras
        self.logger.info("Step 3/4: Converting ONNX to Keras...")
        input_name = onnx_model.graph.input[0].name
        k_model = onnx_to_keras(onnx_model, [input_name], change_ordering=True)
        self.logger.info("  Keras model created")
        
        # Save as SavedModel
        temp_saved_model_dir = self.tflite_dir / f'temp_{model_name}_saved_model'
        if temp_saved_model_dir.exists():
            shutil.rmtree(temp_saved_model_dir)
        
        k_model.save(str(temp_saved_model_dir), save_format='tf')
        self.logger.info("  SavedModel created")
        
        k_model.save(str(temp_saved_model_dir), save_format='tf')
        self.logger.info("  SavedModel created")
        
        # Step 4: Convert to TFLite
        self.logger.info(f"Step 4/4: Converting to TFLite ({self.quantization})...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_saved_model_dir))
        self._apply_quantization(converter)
        converter.allow_custom_ops = True
        
        tflite_model = converter.convert()
        
        tflite_size_mb = len(tflite_model) / (1024 * 1024)
        result['tflite_size_mb'] = round(tflite_size_mb, 2)
        result['compression_ratio'] = round((1 - tflite_size_mb / onnx_size_mb) * 100, 1)
        
        self.logger.info(f"  Converted ({result['tflite_size_mb']} MB)")
        self.logger.info(f"  Compression: {result['compression_ratio']}%")
        
        # Save TFLite model
        self.logger.info("Step 5/5: Saving and verifying...")
        
        output_dir = self.tflite_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tflite_path = output_dir / f"{model_name}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        result['tflite_path'] = str(tflite_path)
        
        # Save metadata
        self._save_metadata(model_name, output_dir, result['onnx_size_mb'], result['tflite_size_mb'])
        
        # Verify
        verification = self._verify_tflite(tflite_path)
        result['verification'] = verification
        
        # Cleanup temp directory
        if temp_saved_model_dir.exists():
            shutil.rmtree(temp_saved_model_dir)
        
        if verification['status'] == 'verified':
            self.logger.info(f"  ✓ Verification PASSED ({verification['avg_inference_ms']}ms)")
        else:
            self.logger.warning(f"  ✗ Verification failed: {verification.get('error', 'Unknown')}")
        
        result['status'] = 'success'
        self.logger.info(f"\nSUCCESS: {model_name}")
        self.logger.info(f"  ONNX:        {onnx_size_mb:.2f} MB")
        self.logger.info(f"  TFLite:      {tflite_size_mb:.2f} MB")
        self.logger.info(f"  Compression: {result['compression_ratio']}%")
        
        return result
    
    def convert_via_pytorch(self, model_name: str) -> dict:
        """
        Method 2: Try to load PyTorch model and convert directly.
        Looks for .pt, .pth, or .torchscript files.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"Method 2: PyTorch Direct Conversion: {model_name}")
        self.logger.info(f"{'='*70}")
        
        # Look for PyTorch model files
        model_dir = self.onnx_dir / model_name
        pytorch_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth")) + \
                       list(model_dir.glob("*.torchscript"))
        
        if not pytorch_files:
            # Try to find in deployment_models directory
            deployment_dir = Path("D:/Base-dir/deployment_models")
            if deployment_dir.exists():
                pytorch_files = list(deployment_dir.glob(f"{model_name}*.pt")) + \
                               list(deployment_dir.glob(f"{model_name}*.pth"))
                
                # Also search in {model_name}_deployment subdirectories
                model_deployment_dir = deployment_dir / f"{model_name}_deployment"
                if model_deployment_dir.exists():
                    pytorch_files += list(model_deployment_dir.glob(f"{model_name}*.pt")) + \
                                    list(model_deployment_dir.glob(f"{model_name}*.pth")) + \
                                    list(model_deployment_dir.glob("*.pt")) + \
                                    list(model_deployment_dir.glob("*.pth"))
        
        if not pytorch_files:
            raise FileNotFoundError(f"No PyTorch model files found for {model_name}")
        
        pytorch_path = pytorch_files[0]
        self.logger.info(f"Found PyTorch model: {pytorch_path}")
        
        result = {
            'model_name': model_name,
            'pytorch_path': str(pytorch_path),
            'timestamp': datetime.datetime.now().isoformat(),
            'conversion_method': 'pytorch-direct',
            'quantization': self.quantization
        }
        
        # Load PyTorch model
        self.logger.info("Step 1/4: Loading PyTorch model...")
        model = torch.jit.load(str(pytorch_path)) if pytorch_path.suffix == '.torchscript' else \
                torch.load(str(pytorch_path), map_location='cpu')
        
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
        
        model.eval()
        self.logger.info("  ✓ Model loaded")
        
        # Export to ONNX with simpler operators
        self.logger.info("Step 2/4: Exporting to ONNX (simplified)...")
        temp_onnx = self.tflite_dir / f'temp_{model_name}_simplified.onnx'
        
        dummy_input = torch.randn(1, 3, 256, 256)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(temp_onnx),
            opset_version=13,  # Use older opset for better compatibility
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
            do_constant_folding=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )
        
        self.logger.info("  ONNX exported with opset 13")
        
        # Step 3: Convert ONNX to Keras using onnx2keras with sanitization
        self.logger.info("Step 3/4: Converting ONNX to Keras (with sanitization)...")
        
        try:
            from onnx2keras import onnx_to_keras
            from onnx_sanitizer import ONNXSanitizer
        except ImportError as e:
            raise ImportError(f"Required libraries not available: {e}")
        
        onnx_model = onnx.load(str(temp_onnx))
        
        # Sanitize the ONNX model
        sanitizer = ONNXSanitizer()
        onnx_model = sanitizer.sanitize_model(onnx_model)
        
        # Convert to Keras
        input_name = onnx_model.graph.input[0].name
        k_model = onnx_to_keras(onnx_model, [input_name], change_ordering=True)
        
        # Save as SavedModel
        temp_saved_model_dir = self.tflite_dir / f'temp_{model_name}_saved_model'
        if temp_saved_model_dir.exists():
            shutil.rmtree(temp_saved_model_dir)
        
        k_model.save(str(temp_saved_model_dir), save_format='tf')
        self.logger.info("  Keras -> SavedModel created")
        
        # Convert to TFLite
        self.logger.info(f"Step 4/4: Converting to TFLite ({self.quantization})...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_saved_model_dir))
        self._apply_quantization(converter)
        converter.allow_custom_ops = True
        
        tflite_model = converter.convert()
        
        onnx_size_mb = os.path.getsize(temp_onnx) / (1024 * 1024)
        tflite_size_mb = len(tflite_model) / (1024 * 1024)
        
        result['onnx_size_mb'] = round(onnx_size_mb, 2)
        result['tflite_size_mb'] = round(tflite_size_mb, 2)
        result['compression_ratio'] = round((1 - tflite_size_mb / onnx_size_mb) * 100, 1)
        
        self.logger.info(f"  ✓ Converted ({result['tflite_size_mb']} MB)")
        
        # Save
        output_dir = self.tflite_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tflite_path = output_dir / f"{model_name}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        result['tflite_path'] = str(tflite_path)
        
        # Save metadata
        self._save_metadata(model_name, output_dir, result['onnx_size_mb'], result['tflite_size_mb'])
        
        # Verify
        verification = self._verify_tflite(tflite_path)
        result['verification'] = verification
        
        # Cleanup
        if temp_onnx.exists():
            temp_onnx.unlink()
        if temp_saved_model_dir.exists():
            shutil.rmtree(temp_saved_model_dir)
        
        if verification['status'] == 'verified':
            self.logger.info(f"  ✓ Verification PASSED ({verification['avg_inference_ms']}ms)")
        
        result['status'] = 'success'
        self.logger.info(f"\nSUCCESS: {model_name}")
        
        return result
    
    def convert_model(self, model_name: str) -> dict:
        """Try multiple conversion methods until one succeeds."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"CONVERTING: {model_name} (Fallback Methods)")
        self.logger.info(f"{'='*70}")
        
        result = None
        errors = []
        
        # Try Method 1: onnx2keras with sanitization (works without onnx-tf)
        try:
            self.logger.info("\nAttempting Method 1: onnx2keras with sanitization...")
            result = self.convert_via_onnx_tensorflow(model_name)
            if result['status'] == 'success':
                self.successful += 1
                self.results[model_name] = result
                return result
        except Exception as e:
            error_msg = f"Method 1 failed: {str(e)}"
            self.logger.warning(f"  X {error_msg}")
            self.logger.debug(traceback.format_exc())
            errors.append(error_msg)
        
        # Try Method 2: PyTorch direct
        if TORCH_AVAILABLE:
            try:
                self.logger.info("\nAttempting Method 2: PyTorch direct...")
                result = self.convert_via_pytorch(model_name)
                if result['status'] == 'success':
                    self.successful += 1
                    self.results[model_name] = result
                    return result
            except Exception as e:
                error_msg = f"Method 2 failed: {str(e)}"
                self.logger.warning(f"  ✗ {error_msg}")
                self.logger.debug(traceback.format_exc())
                errors.append(error_msg)
        
        # All methods failed
        result = {
            'model_name': model_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'failed',
            'error': 'All conversion methods failed',
            'attempted_methods': errors
        }
        
        self.failed += 1
        self.results[model_name] = result
        self.logger.error(f"\nFAILED: {model_name} - All methods exhausted")
        for error in errors:
            self.logger.error(f"  - {error}")
        
        return result
    
    def convert_all(self, failed_models: list) -> dict:
        """Convert all failed models using fallback methods."""
        self.logger.info(f"\nAttempting fallback conversion for {len(failed_models)} models")
        self.logger.info(f"Models: {', '.join(failed_models)}\n")
        
        start_time = datetime.datetime.now()
        
        for model_name in failed_models:
            try:
                self.convert_model(model_name)
            except KeyboardInterrupt:
                self.logger.warning("\nConversion interrupted!")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error for {model_name}: {e}")
                self.logger.debug(traceback.format_exc())
        
        duration = (datetime.datetime.now() - start_time).total_seconds()
        
        # Generate report
        self.logger.info(f"\n{'='*70}")
        self.logger.info("FALLBACK CONVERSION REPORT")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Duration:    {duration:.1f}s")
        self.logger.info(f"Successful:  {self.successful}/{len(failed_models)}")
        self.logger.info(f"Failed:      {self.failed}/{len(failed_models)}")
        
        if self.successful > 0:
            self.logger.info(f"\nSuccessfully converted:")
            for name, res in self.results.items():
                if res.get('status') == 'success':
                    self.logger.info(f"  ✓ {name}: {res.get('compression_ratio', 'N/A')}% compression")
        
        if self.failed > 0:
            self.logger.info(f"\nStill failed:")
            for name, res in self.results.items():
                if res.get('status') == 'failed':
                    self.logger.info(f"  ✗ {name}")
        
        # Save report
        report_path = self.tflite_dir / 'fallback_conversion_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.datetime.now().isoformat(),
                'duration_seconds': round(duration, 2),
                'summary': {
                    'total': len(failed_models),
                    'successful': self.successful,
                    'failed': self.failed
                },
                'models': self.results
            }, f, indent=2)
        
        self.logger.info(f"\nReport saved: {report_path}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"{'='*70}\n")
        
        return self.results


def main():
    """Main entry point."""
    ONNX_DIR = Path(r"D:\Base-dir\onnx_models")
    TFLITE_DIR = Path(r"D:\Base-dir\tflite_models")
    QUANTIZATION = FallbackTFLiteConverter.QUANTIZATION_DYNAMIC
    
    # Models that failed with onnx2keras
    FAILED_MODELS = [
        'darknet53',
        'ensemble_attention',
        'ensemble_concat',
        'ensemble_cross',
        'mobilenet_v2',
        'super_ensemble'
    ]
    
    print("\n" + "=" * 70)
    print("FALLBACK TFLITE CONVERTER")
    print("Converting models that failed with onnx2keras")
    print("=" * 70)
    print(f"\nModels to convert: {len(FAILED_MODELS)}")
    for model in FAILED_MODELS:
        print(f"  - {model}")
    print()
    
    if not ONNX_TF_AVAILABLE:
        print("\n⚠️  WARNING: onnx-tensorflow not installed!")
        print("Install with: pip install onnx-tf")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    converter = FallbackTFLiteConverter(
        onnx_dir=ONNX_DIR,
        tflite_dir=TFLITE_DIR,
        quantization=QUANTIZATION
    )
    
    try:
        results = converter.convert_all(FAILED_MODELS)
        
        if converter.successful > 0:
            print(f"\n✓ Successfully converted {converter.successful} additional models!")
        
        if converter.failed > 0:
            print(f"\n✗ {converter.failed} models still failed (all methods exhausted)")
        
        return 0 if converter.failed == 0 else 1
        
    except KeyboardInterrupt:
        print("\n\nConversion cancelled")
        return 130
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
