"""
Working ONNX to TFLite Converter
Uses onnx2tf for reliable conversion from ONNX to TFLite format.
"""

import os
import json
import subprocess
import datetime
from pathlib import Path
import logging
import tensorflow as tf
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tflite_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ONNXToTFLiteConverter:
    """Converts ONNX models to TFLite using onnx2tf."""
    
    def __init__(self, onnx_models_dir, tflite_output_dir):
        self.onnx_models_dir = Path(onnx_models_dir)
        self.tflite_output_dir = Path(tflite_output_dir)
        self.tflite_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.conversion_report = {
            "conversion_date": datetime.datetime.now().isoformat(),
            "source_format": "ONNX",
            "target_format": "TFLite",
            "models": {},
            "summary": {}
        }
    
    def verify_tflite_model(self, tflite_path):
        """Verify TFLite model can be loaded and run."""
        try:
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test with dummy input
            input_shape = input_details[0]['shape']
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            return {
                'status': 'success',
                'input_shape': input_shape.tolist(),
                'output_shape': output_details[0]['shape'].tolist(),
                'input_dtype': str(input_details[0]['dtype']),
                'output_dtype': str(output_details[0]['dtype']),
                'inference_test': 'passed'
            }
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def convert_model(self, model_name, onnx_path):
        """Convert a single ONNX model to TFLite."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Converting model: {model_name}")
        logger.info(f"{'='*80}")
        
        model_report = {
            'status': 'pending',
            'timestamp': datetime.datetime.now().isoformat(),
            'onnx_source': str(onnx_path)
        }
        
        try:
            # Create output directory for this model
            model_output_dir = self.tflite_output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use onnx2tf to convert
            logger.info("Running onnx2tf conversion...")
            
            cmd = [
                'onnx2tf',
                '-i', str(onnx_path),
                '-o', str(model_output_dir),
                '-osd'  # Output saved model directory
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise Exception(f"onnx2tf failed: {result.stderr}")
            
            logger.info("onnx2tf conversion completed")
            
            # Find the generated TFLite file
            tflite_files = list(model_output_dir.glob("*.tflite"))
            
            if not tflite_files:
                # Try to convert the saved_model to tflite manually
                saved_model_dir = model_output_dir / "saved_model"
                if saved_model_dir.exists():
                    logger.info("Converting SavedModel to TFLite...")
                    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
                    converter.optimizations = []  # No optimization for accuracy
                    tflite_model = converter.convert()
                    
                    tflite_path = model_output_dir / f"{model_name}.tflite"
                    with open(tflite_path, 'wb') as f:
                        f.write(tflite_model)
                    
                    logger.info(f"TFLite model saved to: {tflite_path}")
                else:
                    raise Exception("No TFLite file generated and no SavedModel found")
            else:
                tflite_path = tflite_files[0]
                logger.info(f"TFLite model found: {tflite_path}")
            
            # Verify the model
            verification = self.verify_tflite_model(tflite_path)
            
            # Get file sizes
            onnx_size_mb = round(os.path.getsize(onnx_path) / (1024 * 1024), 2)
            tflite_size_mb = round(os.path.getsize(tflite_path) / (1024 * 1024), 2)
            
            model_report.update({
                'status': 'success',
                'tflite_path': str(tflite_path),
                'onnx_size_mb': onnx_size_mb,
                'tflite_size_mb': tflite_size_mb,
                'size_reduction': f"{round((1 - tflite_size_mb/onnx_size_mb) * 100, 1)}%",
                'verification': verification
            })
            
            logger.info(f"✓ Model {model_name} converted successfully!")
            logger.info(f"  ONNX: {onnx_size_mb} MB -> TFLite: {tflite_size_mb} MB")
            
        except subprocess.TimeoutExpired:
            model_report['status'] = 'failed'
            model_report['error'] = 'Conversion timeout (>5 minutes)'
            logger.error(f"✗ Model {model_name} conversion timed out")
            
        except Exception as e:
            model_report['status'] = 'failed'
            model_report['error'] = str(e)
            logger.error(f"✗ Model {model_name} conversion failed: {str(e)}")
        
        self.conversion_report['models'][model_name] = model_report
        return model_report
    
    def convert_all_models(self):
        """Convert all ONNX models found in the input directory."""
        logger.info("\n" + "="*80)
        logger.info("Starting ONNX to TFLite Conversion")
        logger.info("="*80 + "\n")
        
        # Find all ONNX files in subdirectories
        onnx_files = []
        for model_dir in self.onnx_models_dir.iterdir():
            if model_dir.is_dir():
                onnx_file = model_dir / f"{model_dir.name}.onnx"
                if onnx_file.exists():
                    onnx_files.append((model_dir.name, onnx_file))
        
        if not onnx_files:
            logger.warning(f"No ONNX files found in {self.onnx_models_dir}")
            return
        
        logger.info(f"Found {len(onnx_files)} ONNX models to convert\n")
        
        successful = 0
        failed = 0
        
        for idx, (model_name, onnx_path) in enumerate(onnx_files, 1):
            logger.info(f"\n[{idx}/{len(onnx_files)}] Processing: {model_name}")
            
            result = self.convert_model(model_name, onnx_path)
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
        
        # Generate summary
        self.conversion_report['summary'] = {
            'total_models': len(onnx_files),
            'successful': successful,
            'failed': failed,
            'success_rate': round((successful / len(onnx_files)) * 100, 2) if onnx_files else 0
        }
        
        # Save report
        report_path = self.tflite_output_dir / 'conversion_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.conversion_report, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total models: {len(onnx_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {self.conversion_report['summary']['success_rate']}%")
        logger.info(f"\nReport saved to: {report_path}")
        logger.info("="*80 + "\n")
        
        return self.conversion_report


def main():
    """Main execution function."""
    
    ONNX_MODELS_DIR = r"D:\Base-dir\onnx_models"
    TFLITE_OUTPUT_DIR = r"D:\Base-dir\tflite_models"
    
    print("\n" + "="*80)
    print("ONNX to TFLite Conversion Tool (onnx2tf)")
    print("="*80)
    print(f"\nSource directory: {ONNX_MODELS_DIR}")
    print(f"Output directory: {TFLITE_OUTPUT_DIR}")
    print("\n" + "="*80 + "\n")
    
    converter = ONNXToTFLiteConverter(ONNX_MODELS_DIR, TFLITE_OUTPUT_DIR)
    report = converter.convert_all_models()
    
    if report:
        print("\n" + "="*80)
        print("CONVERSION COMPLETE")
        print("="*80)
        print(f"\nCheck 'tflite_conversion.log' for details")
        print(f"TFLite models saved to: {TFLITE_OUTPUT_DIR}")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
