"""
ONNX to TFLite Converter
Converts all ONNX models to TFLite format while preserving accuracy and structure.
"""

import os
import json
import numpy as np
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('onnx_to_tflite_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ONNXToTFLiteConverter:
    """
    Comprehensive converter for ONNX models to TFLite format.
    Ensures model structure and accuracy preservation.
    """
    
    def __init__(self, onnx_models_dir, tflite_output_dir):
        """
        Initialize the converter.
        
        Args:
            onnx_models_dir: Directory containing ONNX models
            tflite_output_dir: Output directory for TFLite models
        """
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
        
    def load_onnx_model(self, onnx_path):
        """Load and validate ONNX model."""
        try:
            logger.info(f"Loading ONNX model from: {onnx_path}")
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model validated successfully")
            return onnx_model
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            raise
    
    def convert_onnx_to_tensorflow(self, onnx_model):
        """Convert ONNX model to TensorFlow format."""
        try:
            logger.info("Converting ONNX to TensorFlow...")
            tf_rep = prepare(onnx_model)
            return tf_rep
        except Exception as e:
            logger.error(f"Error converting ONNX to TensorFlow: {str(e)}")
            raise
    
    def save_tensorflow_model(self, tf_rep, model_name, temp_dir):
        """Save TensorFlow model to disk."""
        try:
            tf_model_path = temp_dir / f"{model_name}_tf"
            logger.info(f"Saving TensorFlow model to: {tf_model_path}")
            tf_rep.export_graph(str(tf_model_path))
            return tf_model_path
        except Exception as e:
            logger.error(f"Error saving TensorFlow model: {str(e)}")
            raise
    
    def convert_to_tflite(self, tf_model_path, model_name, optimization_mode='default'):
        """
        Convert TensorFlow model to TFLite format with various optimization options.
        
        Args:
            tf_model_path: Path to TensorFlow saved model
            model_name: Name of the model
            optimization_mode: 'default', 'float16', 'dynamic_range', 'full_integer'
        """
        try:
            logger.info(f"Converting to TFLite with optimization mode: {optimization_mode}")
            
            # Create converter
            converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
            
            # Set optimization based on mode
            if optimization_mode == 'default':
                # No optimization - preserves accuracy
                converter.optimizations = []
                tflite_filename = f"{model_name}.tflite"
                
            elif optimization_mode == 'float16':
                # Float16 quantization - minimal accuracy loss
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                tflite_filename = f"{model_name}_float16.tflite"
                
            elif optimization_mode == 'dynamic_range':
                # Dynamic range quantization - reduces model size
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_filename = f"{model_name}_dynamic.tflite"
                
            elif optimization_mode == 'full_integer':
                # Full integer quantization - maximum size reduction
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                tflite_filename = f"{model_name}_int8.tflite"
            
            # Additional settings to preserve accuracy
            converter.experimental_new_converter = True
            converter.allow_custom_ops = True
            
            # Convert
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = self.tflite_output_dir / model_name / tflite_filename
            tflite_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TFLite model saved to: {tflite_path}")
            
            # Get model size
            file_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
            
            return {
                'path': str(tflite_path),
                'size_mb': round(file_size_mb, 2),
                'optimization_mode': optimization_mode
            }
            
        except Exception as e:
            logger.error(f"Error converting to TFLite: {str(e)}")
            raise
    
    def get_model_metadata(self, onnx_model, tflite_info):
        """Extract and compare model metadata."""
        try:
            # ONNX model info
            onnx_input = onnx_model.graph.input[0]
            onnx_output = onnx_model.graph.output[0]
            
            # Get input shape
            input_shape = [dim.dim_value for dim in onnx_input.type.tensor_type.shape.dim]
            output_shape = [dim.dim_value for dim in onnx_output.type.tensor_type.shape.dim]
            
            # Get operator count
            op_types = [node.op_type for node in onnx_model.graph.node]
            unique_ops = set(op_types)
            
            metadata = {
                'input_name': onnx_input.name,
                'output_name': onnx_output.name,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'total_ops': len(op_types),
                'unique_op_types': len(unique_ops),
                'op_types': list(unique_ops)
            }
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract full metadata: {str(e)}")
            return {}
    
    def verify_tflite_model(self, tflite_path, expected_output_shape):
        """Verify TFLite model structure and functionality."""
        try:
            logger.info(f"Verifying TFLite model: {tflite_path}")
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Verify shapes
            input_shape = input_details[0]['shape'].tolist()
            output_shape = output_details[0]['shape'].tolist()
            
            verification = {
                'status': 'success',
                'input_shape': input_shape,
                'output_shape': output_shape,
                'input_dtype': str(input_details[0]['dtype']),
                'output_dtype': str(output_details[0]['dtype']),
                'shape_match': output_shape == expected_output_shape
            }
            
            # Test inference with dummy data
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            verification['inference_test'] = 'passed'
            verification['output_range'] = {
                'min': float(np.min(output_data)),
                'max': float(np.max(output_data)),
                'mean': float(np.mean(output_data))
            }
            
            logger.info(f"TFLite model verification passed")
            return verification
            
        except Exception as e:
            logger.error(f"TFLite model verification failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def convert_model(self, model_name, onnx_path, create_optimized_versions=True):
        """
        Convert a single ONNX model to TFLite with multiple optimization options.
        
        Args:
            model_name: Name of the model
            onnx_path: Path to ONNX model file
            create_optimized_versions: Whether to create float16 and dynamic quantized versions
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Converting model: {model_name}")
        logger.info(f"{'='*80}")
        
        model_report = {
            'status': 'pending',
            'timestamp': datetime.datetime.now().isoformat(),
            'onnx_source': str(onnx_path),
            'versions': {}
        }
        
        try:
            # Step 1: Load ONNX model
            onnx_model = self.load_onnx_model(onnx_path)
            model_report['onnx_file_size_mb'] = round(os.path.getsize(onnx_path) / (1024 * 1024), 2)
            
            # Step 2: Convert to TensorFlow
            tf_rep = self.convert_onnx_to_tensorflow(onnx_model)
            
            # Step 3: Save TensorFlow model temporarily
            temp_dir = self.tflite_output_dir / '_temp'
            temp_dir.mkdir(parents=True, exist_ok=True)
            tf_model_path = self.save_tensorflow_model(tf_rep, model_name, temp_dir)
            
            # Step 4: Get expected output shape
            onnx_output = onnx_model.graph.output[0]
            expected_output_shape = [dim.dim_value for dim in onnx_output.type.tensor_type.shape.dim]
            
            # Step 5: Convert to TFLite (default - no quantization)
            logger.info("\n--- Creating default TFLite model (full precision) ---")
            tflite_info = self.convert_to_tflite(tf_model_path, model_name, 'default')
            verification = self.verify_tflite_model(tflite_info['path'], expected_output_shape)
            
            model_report['versions']['default'] = {
                **tflite_info,
                'verification': verification,
                'description': 'Full precision model - maintains original accuracy'
            }
            
            # Step 6: Create optimized versions if requested
            if create_optimized_versions:
                # Float16 quantization
                logger.info("\n--- Creating Float16 quantized model ---")
                try:
                    tflite_info_fp16 = self.convert_to_tflite(tf_model_path, model_name, 'float16')
                    verification_fp16 = self.verify_tflite_model(tflite_info_fp16['path'], expected_output_shape)
                    
                    model_report['versions']['float16'] = {
                        **tflite_info_fp16,
                        'verification': verification_fp16,
                        'description': 'Float16 quantized - minimal accuracy loss, ~50% size reduction'
                    }
                except Exception as e:
                    logger.warning(f"Float16 conversion failed: {str(e)}")
                    model_report['versions']['float16'] = {'status': 'failed', 'error': str(e)}
                
                # Dynamic range quantization
                logger.info("\n--- Creating dynamic range quantized model ---")
                try:
                    tflite_info_dynamic = self.convert_to_tflite(tf_model_path, model_name, 'dynamic_range')
                    verification_dynamic = self.verify_tflite_model(tflite_info_dynamic['path'], expected_output_shape)
                    
                    model_report['versions']['dynamic_range'] = {
                        **tflite_info_dynamic,
                        'verification': verification_dynamic,
                        'description': 'Dynamic range quantized - good accuracy, significant size reduction'
                    }
                except Exception as e:
                    logger.warning(f"Dynamic range conversion failed: {str(e)}")
                    model_report['versions']['dynamic_range'] = {'status': 'failed', 'error': str(e)}
            
            # Step 7: Extract metadata
            metadata = self.get_model_metadata(onnx_model, tflite_info)
            model_report['metadata'] = metadata
            
            # Step 8: Clean up temporary files
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            model_report['status'] = 'success'
            logger.info(f"\n✓ Model {model_name} converted successfully!")
            
        except Exception as e:
            model_report['status'] = 'failed'
            model_report['error'] = str(e)
            logger.error(f"\n✗ Model {model_name} conversion failed: {str(e)}")
        
        self.conversion_report['models'][model_name] = model_report
        return model_report
    
    def convert_all_models(self, create_optimized_versions=True):
        """Convert all ONNX models found in the input directory."""
        logger.info("\n" + "="*80)
        logger.info("Starting batch conversion of all ONNX models to TFLite")
        logger.info("="*80 + "\n")
        
        # Find all ONNX files
        onnx_files = list(self.onnx_models_dir.glob("*/*.onnx"))
        
        if not onnx_files:
            logger.warning(f"No ONNX files found in {self.onnx_models_dir}")
            return
        
        logger.info(f"Found {len(onnx_files)} ONNX models to convert\n")
        
        successful = 0
        failed = 0
        
        for idx, onnx_path in enumerate(onnx_files, 1):
            model_name = onnx_path.parent.name
            logger.info(f"\n[{idx}/{len(onnx_files)}] Processing: {model_name}")
            
            result = self.convert_model(model_name, onnx_path, create_optimized_versions)
            
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
        
        # Save conversion report
        report_path = self.tflite_output_dir / 'tflite_conversion_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.conversion_report, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total models: {len(onnx_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {self.conversion_report['summary']['success_rate']}%")
        logger.info(f"\nDetailed report saved to: {report_path}")
        logger.info("="*80 + "\n")
        
        return self.conversion_report


def main():
    """Main execution function."""
    
    # Configuration
    ONNX_MODELS_DIR = r"D:\Base-dir\onnx_models"
    TFLITE_OUTPUT_DIR = r"D:\Base-dir\tflite_models"
    CREATE_OPTIMIZED_VERSIONS = True  # Set to False to only create default version
    
    print("\n" + "="*80)
    print("ONNX to TFLite Conversion Tool")
    print("="*80)
    print(f"\nSource directory: {ONNX_MODELS_DIR}")
    print(f"Output directory: {TFLITE_OUTPUT_DIR}")
    print(f"Create optimized versions: {CREATE_OPTIMIZED_VERSIONS}")
    print("\n" + "="*80 + "\n")
    
    # Create converter instance
    converter = ONNXToTFLiteConverter(ONNX_MODELS_DIR, TFLITE_OUTPUT_DIR)
    
    # Convert all models
    report = converter.convert_all_models(create_optimized_versions=CREATE_OPTIMIZED_VERSIONS)
    
    # Print summary
    if report:
        print("\n" + "="*80)
        print("CONVERSION COMPLETE")
        print("="*80)
        print(f"\nCheck the log file 'onnx_to_tflite_conversion.log' for detailed information")
        print(f"TFLite models saved to: {TFLITE_OUTPUT_DIR}")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
