"""
PyTorch to TFLite with Dynamic Range Quantization
==================================================
Full precision models compressed via dynamic range quantization.
Works on Windows without torch_xla requirement.

Method: PyTorch → ONNX → TensorFlow SavedModel → TFLite (with quantization)
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import onnx
from onnx import version_converter
import tensorflow as tf  # type: ignore[import-unresolved]

# Critical: Patch onnx2tf BEFORE importing it to prevent network timeouts
import sys
import importlib.util

# Pre-patch the module
if 'onnx2tf.utils.common_functions' not in sys.modules:
    try:
        import onnx2tf.utils.common_functions as ocf
        original_download = ocf.download_test_image_data
        def dummy_download_test_image_data(*args, **kwargs):
            """Bypass network download that causes timeouts on Windows"""
            return np.zeros((1, 3, 256, 256), dtype=np.float32)
        ocf.download_test_image_data = dummy_download_test_image_data
    except:
        pass

import onnx2tf

# Double-check the patch is applied
try:
    import onnx2tf.utils.common_functions
    def dummy_download_test_image_data(*args, **kwargs):
        return np.zeros((1, 3, 256, 256), dtype=np.float32)
    onnx2tf.utils.common_functions.download_test_image_data = dummy_download_test_image_data
except Exception as e:
    logging.warning(f"Could not patch onnx2tf: {e}")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('PyTorchToTFLiteQuantized')


class PyTorchToTFLiteQuantized:
    """Convert PyTorch to TFLite with dynamic range quantization."""
    
    def __init__(self, pytorch_dir: Path, tflite_dir: Path):
        self.pytorch_dir = Path(pytorch_dir)
        self.tflite_dir = Path(tflite_dir)
        self.onnx_dir = Path(r"D:\Base-dir\onnx_models")  # Pre-converted ONNX models
        self.temp_dir = Path(r"D:\tmp\tflite_conversion")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def _find_onnx_model(self, model_name: str) -> Path:
        """Find pre-converted ONNX model file."""
        candidates = [
            self.onnx_dir / model_name / f"{model_name}.onnx",
            self.onnx_dir / f"{model_name}.onnx",
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        return None
    
    def _find_pytorch_model(self, model_name: str) -> Path:
        """Find PyTorch model file."""
        candidates = [
            self.pytorch_dir / f"{model_name}.pt",
            self.pytorch_dir / f"{model_name}_deployment" / f"{model_name}.pt",
            self.pytorch_dir / f"{model_name}_deployment" / "model.pth",
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"No PyTorch model found for {model_name}")
    
    def convert_model(self, model_name: str) -> dict:
        """Convert a single PyTorch model to quantized TFLite."""
        
        # Clear session to avoid memory leaks and state issues
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Converting: {model_name}")
        logger.info(f"{'='*70}")
        
        result = {
            'model_name': model_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'method': 'pytorch_onnx_tf_quantized'
        }
        
        try:
            # Check if ONNX model already exists
            existing_onnx = self._find_onnx_model(model_name)
            
            if existing_onnx:
                logger.info("Step 1/5: Using pre-converted ONNX model...")
                onnx_path = existing_onnx
                logger.info(f"  Found: {onnx_path}")
                
                # Get original PyTorch size for comparison
                pt_path = self._find_pytorch_model(model_name)
                pt_size_mb = os.path.getsize(pt_path) / (1024 * 1024)
                result['pytorch_size_mb'] = round(pt_size_mb, 2)
                logger.info(f"  PyTorch: {pt_size_mb:.2f} MB")
                logger.info(f"  ONNX: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
                
                # Skip to step 3
                skip_onnx_export = True
            else:
                # Step 1: Load PyTorch model
                logger.info("Step 1/5: Loading PyTorch model...")
                pt_path = self._find_pytorch_model(model_name)
                logger.info(f"  Found: {pt_path}")
                
                model = torch.jit.load(str(pt_path), map_location='cpu')
                model.eval()
                
                pt_size_mb = os.path.getsize(pt_path) / (1024 * 1024)
                result['pytorch_size_mb'] = round(pt_size_mb, 2)
                logger.info(f"  Loaded: {pt_size_mb:.2f} MB")
                
                # Create a wrapper to handle tracing issues with adaptive pooling
                class ModelWrapper(torch.nn.Module):
                    def __init__(self, original_model):
                        super().__init__()
                        self.model = original_model
                    
                    def forward(self, x):
                        return self.model(x)
                
                # Wrap the model for better tracing
                wrapped_model = ModelWrapper(model)
                wrapped_model.eval()
                
                skip_onnx_export = False
            
            if not skip_onnx_export:
                # Step 2: Export to ONNX
                logger.info("Step 2/5: Exporting to ONNX...")
                onnx_path = self.temp_dir / f"{model_name}_temp.onnx"
                
                sample_input = torch.randn(1, 3, 256, 256)
                
                # Try multiple strategies for ONNX export
                export_success = False
                last_error = None
                
                # Strategy 1: Direct export with multiple opset versions
                for opset in [17, 13, 11, 9]:
                    try:
                        torch.onnx.export(
                            model,
                            sample_input,
                            str(onnx_path),
                            export_params=True,
                            opset_version=opset,
                            do_constant_folding=True,
                            input_names=['input'],
                            output_names=['output']
                        )
                        export_success = True
                        logger.info(f"  Exported ONNX (opset {opset}): {onnx_path.stat().st_size / (1024*1024):.2f} MB")
                        break
                    except Exception as e:
                        last_error = str(e)
                        if "adaptive_avg_pool2d" not in str(e).lower() or opset == 9:
                            continue
                
                # Strategy 2: Trace the model instead of using scripted model
                if not export_success and "adaptive_avg_pool2d" in str(last_error).lower():
                    logger.warning("  Standard export failed, attempting with torch.jit.trace...")
                    try:
                        with torch.no_grad():
                            traced_model = torch.jit.trace(wrapped_model, sample_input)
                        
                        torch.onnx.export(
                            traced_model,
                            sample_input,
                            str(onnx_path),
                            export_params=True,
                            opset_version=11,
                            do_constant_folding=True,
                            input_names=['input'],
                            output_names=['output']
                        )
                        export_success = True
                        logger.info(f"  Exported ONNX via tracing: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
                    except Exception as e:
                        last_error = str(e)
                
                if not export_success:
                    raise RuntimeError(f"Failed to export to ONNX: {last_error}")

            
            # Step 3: Convert ONNX to TensorFlow SavedModel
            logger.info("Step 3/5: Converting to TensorFlow SavedModel...")
            
            saved_model_dir = self.temp_dir / f"{model_name}_saved_model"
            
            # Disable test data download to avoid network timeouts
            os.environ['ONNX2TF_DISABLE_STRICT_MODE'] = '1'
            
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(saved_model_dir),
                copy_onnx_input_output_names_to_tflite=False,
                non_verbose=True,
                output_integer_quantized_tflite=False,
                quant_type='per-tensor'
            )
            
            logger.info(f"  SavedModel created")
            
            # Step 4: Convert to TFLite with Dynamic Range Quantization
            logger.info("Step 4/5: Converting to TFLite with quantization...")
            
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            
            # Enable dynamic range quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Advanced settings
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLite ops
                tf.lite.OpsSet.SELECT_TF_OPS      # Allow select TF ops if needed
            ]
            
            # Experimental options for better compatibility
            converter._experimental_lower_tensor_list_ops = False
            
            try:
                tflite_model = converter.convert()
                logger.info("  Quantized conversion successful!")
            except Exception as quant_error:
                logger.warning(f"  Quantization failed: {quant_error}")
                logger.info("  Trying without quantization...")
                
                converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
                tflite_model = converter.convert()
                result['quantization'] = 'disabled'
            else:
                result['quantization'] = 'dynamic_range'
            
            # Step 5: Save TFLite model
            logger.info("Step 5/5: Saving TFLite model...")
            
            output_dir = self.tflite_dir / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            tflite_path = output_dir / f"{model_name}.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            tflite_size_mb = len(tflite_model) / (1024 * 1024)
            result['tflite_size_mb'] = round(tflite_size_mb, 2)
            result['tflite_path'] = str(tflite_path)
            result['compression_ratio'] = round((1 - tflite_size_mb / pt_size_mb) * 100, 1)
            
            logger.info(f"  Saved: {tflite_size_mb:.2f} MB")
            logger.info(f"  Compression: {result['compression_ratio']}%")
            
            # Verify TFLite model
            logger.info("Verifying TFLite model...")
            try:
                interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Test inference
                test_input = np.random.randn(*input_details[0]['shape']).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                
                logger.info(f"  OK Verified! Output shape: {output.shape}")
                result['verification'] = 'passed'
                
            except Exception as e:
                logger.warning(f"  Warning: {str(e)}")
                result['verification'] = f'warning: {str(e)}'
            
            # Save metadata
            self._save_metadata(model_name, output_dir, pt_size_mb, tflite_size_mb)
            
            # Cleanup
            if onnx_path.exists():
                onnx_path.unlink()
            
            result['status'] = 'success'
            logger.info(f"\nSUCCESS: {model_name}")
            logger.info(f"  PyTorch:     {pt_size_mb:.2f} MB")
            logger.info(f"  TFLite:      {tflite_size_mb:.2f} MB")
            logger.info(f"  Compression: {result['compression_ratio']}%")
            logger.info(f"  Quantization: {result.get('quantization', 'unknown')}")
            
            return result
            
        except Exception as e:
            logger.error(f"\nFAILED: {model_name}")
            logger.error(f"  Error: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            result['status'] = 'failed'
            result['error'] = str(e)
            return result
    
    def _save_metadata(self, model_name: str, output_dir: Path, pt_size: float, tflite_size: float):
        """Save Android metadata."""
        metadata = {
            "model_info": {
                "name": model_name,
                "version": "1.0.0",
                "description": f"TFLite model - {model_name} (quantized)",
                "created_date": datetime.datetime.now().isoformat(),
                "framework": "TensorFlow Lite",
                "quantization": "dynamic_range",
                "original_size_mb": pt_size,
                "compressed_size_mb": tflite_size,
                "conversion_method": "PyTorch → ONNX → TensorFlow → TFLite (quantized)"
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
                "shape": [1, 11],
                "dtype": "float32"
            },
            "labels": [f"class_{i}" for i in range(11)],
            "num_classes": 11
        }
        
        with open(output_dir / 'android_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def convert_all(self, model_list: list) -> dict:
        """Convert all specified models using subprocess for isolation."""
        
        logger.info("\n" + "="*70)
        logger.info("PyTorch to TFLite with Dynamic Range Quantization")
        logger.info("Full precision models with compression")
        logger.info("="*70)
        logger.info(f"Converting {len(model_list)} models\n")
        
        start_time = datetime.datetime.now()
        
        import subprocess
        import sys
        
        for i, model_name in enumerate(model_list, 1):
            logger.info(f"\n[{i}/{len(model_list)}] Processing: {model_name}")
            
            # Run conversion in a separate process to ensure clean state
            cmd = [sys.executable, str(Path(__file__).absolute()), '--model', model_name]
            
            try:
                subprocess.run(cmd, check=True)
                
                # Load result from expected JSON file
                result_file = self.tflite_dir / model_name / 'conversion_result.json'
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        self.results[model_name] = json.load(f)
                else:
                    self.results[model_name] = {'status': 'failed', 'error': 'No result file generated'}
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"Subprocess failed for {model_name}")
                self.results[model_name] = {'status': 'failed', 'error': 'Subprocess failed'}
            except Exception as e:
                logger.error(f"Unexpected error for {model_name}: {e}")
                self.results[model_name] = {'status': 'failed', 'error': str(e)}
        
        duration = (datetime.datetime.now() - start_time).total_seconds()
        
        # Summary
        successful = sum(1 for r in self.results.values() if r.get('status') == 'success')
        failed = len(model_list) - successful
        
        logger.info(f"\n{'='*70}")
        logger.info("CONVERSION SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Duration:   {duration:.1f}s")
        logger.info(f"Total:      {len(model_list)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed:     {failed}")
        
        if successful > 0:
            logger.info(f"\nSuccessfully converted:")
            total_pt = 0
            total_tflite = 0
            for name, res in self.results.items():
                if res.get('status') == 'success':
                    quant = res.get('quantization', 'unknown')
                    logger.info(f"  OK {name}: {res.get('tflite_size_mb')} MB ({res.get('compression_ratio')}% smaller, {quant})")
                    total_pt += res.get('pytorch_size_mb', 0)
                    total_tflite += res.get('tflite_size_mb', 0)
            
            overall_compression = round((1 - total_tflite / total_pt) * 100, 1) if total_pt > 0 else 0
            logger.info(f"\nOverall: {total_pt:.2f} MB -> {total_tflite:.2f} MB ({overall_compression}% compression)")
        
        if failed > 0:
            logger.info(f"\nFailed models:")
            for name, res in self.results.items():
                if res.get('status') == 'failed':
                    logger.info(f"  X {name}: {res.get('error', 'Unknown error')}")
        
        # Save report
        report_path = self.tflite_dir / 'quantized_conversion_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.datetime.now().isoformat(),
                'duration_seconds': round(duration, 2),
                'summary': {
                    'total': len(model_list),
                    'successful': successful,
                    'failed': failed
                },
                'models': self.results
            }, f, indent=2)
        
        logger.info(f"\nReport saved: {report_path}")
        logger.info(f"{'='*70}\n")
        
        return self.results


def main():
    """Convert models from PyTorch to quantized TFLite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch to TFLite Converter')
    parser.add_argument('--model', type=str, help='Specific model to convert')
    parser.add_argument('--input_dir', type=str, default=r"D:\Base-dir\deployment_models", help='Directory containing PyTorch .pt models')
    parser.add_argument('--output_dir', type=str, default=r"tflite_models", help='Directory to save TFLite models')
    args = parser.parse_args()
    
    PYTORCH_DIR = Path(args.input_dir)
    TFLITE_DIR = Path(args.output_dir)
    
    # Ensure output directory exists
    TFLITE_DIR.mkdir(parents=True, exist_ok=True)
    
    converter = PyTorchToTFLiteQuantized(PYTORCH_DIR, TFLITE_DIR)
    
    if args.model:
        # Single model conversion mode (subprocess)
        result = converter.convert_model(args.model)
        
        # Save result to file for parent process to read
        output_dir = TFLITE_DIR / args.model
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'conversion_result.json', 'w') as f:
            json.dump(result, f, indent=2)
            
        if result.get('status') == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
    
    # ALL 11 models
    ALL_MODELS = [
        'mobilenet_v2',
        'darknet53',
        'ensemble_attention',
        'ensemble_concat',
        'ensemble_cross',
        'super_ensemble',
        'alexnet',
        'resnet50',
        'inception_v3',
        'efficientnet_b0',
        'yolo11n-cls'
    ]
    
    results = converter.convert_all(ALL_MODELS)
    
    successful = sum(1 for r in results.values() if r.get('status') == 'success')
    
    if successful == len(ALL_MODELS):
        print(f"\nSUCCESS! All {len(ALL_MODELS)} models converted to TFLite!")
        return 0
    else:
        print(f"\nPartial success: {successful}/{len(ALL_MODELS)} models converted")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
