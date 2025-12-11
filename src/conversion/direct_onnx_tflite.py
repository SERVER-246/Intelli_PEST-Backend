"""
Direct ONNX to TFLite Converter using tf2onnx and TensorFlow
Simple, reliable conversion without external tools.
"""

import os
import json
import datetime
from pathlib import Path
import logging
import tensorflow as tf
import onnx
import numpy as np
import onnxruntime as ort

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tflite_conversion.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimpleONNXToTFLiteConverter:
    """Simple converter using ONNX Runtime and TensorFlow."""
    
    def __init__(self, onnx_models_dir, tflite_output_dir):
        self.onnx_models_dir = Path(onnx_models_dir)
        self.tflite_output_dir = Path(tflite_output_dir)
        self.tflite_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.conversion_report = {
            "conversion_date": datetime.datetime.now().isoformat(),
            "models": {},
            "summary": {}
        }
    
    def convert_model(self, model_name, onnx_path):
        """Convert single ONNX model to TFLite."""
        logger.info(f"\n[Converting: {model_name}]")
        
        model_report = {
            'status': 'pending',
            'onnx_source': str(onnx_path)
        }
        
        try:
            # Load ONNX model to get input/output info
            onnx_model = onnx.load(str(onnx_path))
            session = ort.InferenceSession(str(onnx_path))
            
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            input_shape = input_info.shape
            input_name = input_info.name
            output_name = output_info.name
            
            logger.info(f"  Input: {input_name} {input_shape}")
            logger.info(f"  Output: {output_name}")
            
            # Create a TensorFlow model that wraps ONNX inference
            class ONNXModel(tf.Module):
                def __init__(self, onnx_path):
                    super().__init__()
                    self.session = ort.InferenceSession(str(onnx_path))
                    self.input_name = self.session.get_inputs()[0].name
                    
                @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3, 256, 256], dtype=tf.float32)])
                def __call__(self, x):
                    # Run ONNX inference in eager mode
                    @tf.py_function(Tout=tf.float32)
                    def onnx_infer(inputs):
                        result = self.session.run(None, {self.input_name: inputs.numpy()})
                        return result[0]
                    
                    return onnx_infer(x)
            
            # Actually, let's use a simpler approach - just use TFLite's ONNX import
            # TensorFlow doesn't have direct ONNX import, so let's convert via saved_model
            
            # Try using onnx2keras which we have installed
            from onnx2keras import onnx_to_keras
            import keras
            
            logger.info("  Converting ONNX to Keras...")
            k_model = onnx_to_keras(onnx_model, [input_name])
            
            # Save as Keras model
            keras_path = self.tflite_output_dir / model_name / "keras_model.h5"
            keras_path.parent.mkdir(parents=True, exist_ok=True)
            k_model.save(str(keras_path))
            
            logger.info("  Converting Keras to TFLite...")
            converter = tf.lite.TFLiteConverter.from_keras_model(k_model)
            converter.optimizations = []  # No quantization
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = self.tflite_output_dir / model_name / f"{model_name}.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            # Test the model
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            
            onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
            tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
            
            model_report.update({
                'status': 'success',
                'tflite_path': str(tflite_path),
                'onnx_size_mb': round(onnx_size, 2),
                'tflite_size_mb': round(tflite_size, 2)
            })
            
            logger.info(f"  SUCCESS: {tflite_size:.2f} MB TFLite model created")
            
        except Exception as e:
            model_report['status'] = 'failed'
            model_report['error'] = str(e)
            logger.error(f"  FAILED: {str(e)}")
        
        self.conversion_report['models'][model_name] = model_report
        return model_report
    
    def convert_all(self):
        """Convert all ONNX models."""
        logger.info("="*80)
        logger.info("ONNX to TFLite Conversion (via Keras)")
        logger.info("="*80)
        
        # Find all ONNX models
        onnx_files = []
        for model_dir in self.onnx_models_dir.iterdir():
            if model_dir.is_dir():
                onnx_file = model_dir / f"{model_dir.name}.onnx"
                if onnx_file.exists():
                    onnx_files.append((model_dir.name, onnx_file))
        
        if not onnx_files:
            logger.error(f"No ONNX files found in {self.onnx_models_dir}")
            return None
        
        logger.info(f"Found {len(onnx_files)} ONNX models\n")
        
        successful = 0
        failed = 0
        
        for idx, (model_name, onnx_path) in enumerate(onnx_files, 1):
            logger.info(f"[{idx}/{len(onnx_files)}] {model_name}")
            result = self.convert_model(model_name, onnx_path)
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
        
        self.conversion_report['summary'] = {
            'total': len(onnx_files),
            'successful': successful,
            'failed': failed,
            'success_rate': round((successful / len(onnx_files)) * 100, 2)
        }
        
        # Save report
        report_path = self.tflite_output_dir / 'conversion_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.conversion_report, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        logger.info(f"Total: {len(onnx_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {self.conversion_report['summary']['success_rate']}%")
        logger.info("="*80)
        
        return self.conversion_report


def main():
    ONNX_DIR = r"D:\Base-dir\onnx_models"
    TFLITE_DIR = r"D:\Base-dir\tflite_models"
    
    converter = SimpleONNXToTFLiteConverter(ONNX_DIR, TFLITE_DIR)
    converter.convert_all()


if __name__ == "__main__":
    main()
