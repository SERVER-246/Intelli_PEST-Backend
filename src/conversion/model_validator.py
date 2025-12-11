"""
TFLite Model Accuracy Validation Script
Compares ONNX and TFLite model outputs to ensure accuracy is preserved.
"""

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tflite_accuracy_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelAccuracyValidator:
    """
    Validates that TFLite models maintain accuracy compared to original ONNX models.
    """
    
    def __init__(self, onnx_models_dir, tflite_models_dir):
        """
        Initialize validator.
        
        Args:
            onnx_models_dir: Directory containing ONNX models
            tflite_models_dir: Directory containing TFLite models
        """
        self.onnx_models_dir = Path(onnx_models_dir)
        self.tflite_models_dir = Path(tflite_models_dir)
        self.validation_results = {
            'models': {},
            'summary': {}
        }
    
    def generate_test_inputs(self, input_shape, num_samples=10):
        """
        Generate test inputs for validation.
        
        Args:
            input_shape: Shape of input tensor
            num_samples: Number of test samples to generate
        
        Returns:
            List of test input arrays
        """
        test_inputs = []
        
        # Generate random inputs with different distributions
        for i in range(num_samples):
            if i < 3:
                # Normal distribution
                test_input = np.random.randn(*input_shape).astype(np.float32)
            elif i < 6:
                # Uniform distribution [0, 1]
                test_input = np.random.rand(*input_shape).astype(np.float32)
            else:
                # Edge cases
                if i == 6:
                    test_input = np.zeros(input_shape, dtype=np.float32)
                elif i == 7:
                    test_input = np.ones(input_shape, dtype=np.float32)
                elif i == 8:
                    test_input = np.full(input_shape, 0.5, dtype=np.float32)
                else:
                    test_input = np.random.randn(*input_shape).astype(np.float32) * 0.1
            
            test_inputs.append(test_input)
        
        return test_inputs
    
    def run_onnx_inference(self, onnx_path, test_inputs):
        """
        Run inference using ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            test_inputs: List of test input arrays
        
        Returns:
            List of output arrays
        """
        try:
            logger.info(f"Running ONNX inference for: {onnx_path.name}")
            
            # Create ONNX Runtime session
            sess = ort.InferenceSession(str(onnx_path))
            input_name = sess.get_inputs()[0].name
            
            outputs = []
            for test_input in test_inputs:
                output = sess.run(None, {input_name: test_input})[0]
                outputs.append(output)
            
            return outputs
            
        except Exception as e:
            logger.error(f"ONNX inference failed: {str(e)}")
            raise
    
    def run_tflite_inference(self, tflite_path, test_inputs):
        """
        Run inference using TFLite model.
        
        Args:
            tflite_path: Path to TFLite model
            test_inputs: List of test input arrays
        
        Returns:
            List of output arrays
        """
        try:
            logger.info(f"Running TFLite inference for: {tflite_path.name}")
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            outputs = []
            for test_input in test_inputs:
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], test_input)
                
                # Run inference
                interpreter.invoke()
                
                # Get output
                output = interpreter.get_tensor(output_details[0]['index'])
                outputs.append(output)
            
            return outputs
            
        except Exception as e:
            logger.error(f"TFLite inference failed: {str(e)}")
            raise
    
    def calculate_accuracy_metrics(self, onnx_outputs, tflite_outputs):
        """
        Calculate various accuracy metrics between ONNX and TFLite outputs.
        
        Args:
            onnx_outputs: List of ONNX output arrays
            tflite_outputs: List of TFLite output arrays
        
        Returns:
            Dictionary of accuracy metrics
        """
        metrics = {
            'max_absolute_error': [],
            'mean_absolute_error': [],
            'mean_squared_error': [],
            'relative_error': [],
            'cosine_similarity': []
        }
        
        for onnx_out, tflite_out in zip(onnx_outputs, tflite_outputs):
            # Ensure same shape
            if onnx_out.shape != tflite_out.shape:
                logger.warning(f"Shape mismatch: ONNX {onnx_out.shape} vs TFLite {tflite_out.shape}")
                continue
            
            # Max absolute error
            max_abs_error = np.max(np.abs(onnx_out - tflite_out))
            metrics['max_absolute_error'].append(float(max_abs_error))
            
            # Mean absolute error
            mae = np.mean(np.abs(onnx_out - tflite_out))
            metrics['mean_absolute_error'].append(float(mae))
            
            # Mean squared error
            mse = np.mean((onnx_out - tflite_out) ** 2)
            metrics['mean_squared_error'].append(float(mse))
            
            # Relative error
            epsilon = 1e-8
            relative_error = np.mean(np.abs(onnx_out - tflite_out) / (np.abs(onnx_out) + epsilon))
            metrics['relative_error'].append(float(relative_error))
            
            # Cosine similarity
            onnx_flat = onnx_out.flatten()
            tflite_flat = tflite_out.flatten()
            cosine_sim = np.dot(onnx_flat, tflite_flat) / (
                np.linalg.norm(onnx_flat) * np.linalg.norm(tflite_flat) + epsilon
            )
            metrics['cosine_similarity'].append(float(cosine_sim))
        
        # Calculate statistics
        summary_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                summary_metrics[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return summary_metrics
    
    def validate_model(self, model_name, onnx_path, tflite_path, num_test_samples=10):
        """
        Validate a single model by comparing ONNX and TFLite outputs.
        
        Args:
            model_name: Name of the model
            onnx_path: Path to ONNX model
            tflite_path: Path to TFLite model
            num_test_samples: Number of test samples to use
        
        Returns:
            Validation results dictionary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Validating model: {model_name}")
        logger.info(f"{'='*80}")
        
        validation_result = {
            'model_name': model_name,
            'status': 'pending',
            'onnx_path': str(onnx_path),
            'tflite_path': str(tflite_path)
        }
        
        try:
            # Get input shape from ONNX model
            onnx_model = onnx.load(str(onnx_path))
            input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
            logger.info(f"Input shape: {input_shape}")
            
            # Generate test inputs
            test_inputs = self.generate_test_inputs(input_shape, num_test_samples)
            logger.info(f"Generated {len(test_inputs)} test samples")
            
            # Run ONNX inference
            onnx_outputs = self.run_onnx_inference(onnx_path, test_inputs)
            
            # Run TFLite inference
            tflite_outputs = self.run_tflite_inference(tflite_path, test_inputs)
            
            # Calculate accuracy metrics
            metrics = self.calculate_accuracy_metrics(onnx_outputs, tflite_outputs)
            validation_result['accuracy_metrics'] = metrics
            
            # Determine if validation passed
            # Criteria: Mean absolute error < 1e-3 and cosine similarity > 0.999
            mae_mean = metrics['mean_absolute_error']['mean']
            cosine_sim_mean = metrics['cosine_similarity']['mean']
            
            if mae_mean < 1e-3 and cosine_sim_mean > 0.999:
                validation_result['status'] = 'passed'
                validation_result['accuracy_preserved'] = True
                logger.info(f"✓ Validation PASSED - Accuracy preserved")
            elif mae_mean < 1e-2 and cosine_sim_mean > 0.99:
                validation_result['status'] = 'passed_with_minor_differences'
                validation_result['accuracy_preserved'] = True
                logger.info(f"✓ Validation PASSED with minor differences - Acceptable accuracy")
            else:
                validation_result['status'] = 'failed'
                validation_result['accuracy_preserved'] = False
                logger.warning(f"✗ Validation FAILED - Significant accuracy differences detected")
            
            # Add summary statistics
            validation_result['summary'] = {
                'mean_absolute_error': mae_mean,
                'cosine_similarity': cosine_sim_mean,
                'max_absolute_error': metrics['max_absolute_error']['max']
            }
            
            logger.info(f"Mean Absolute Error: {mae_mean:.6e}")
            logger.info(f"Cosine Similarity: {cosine_sim_mean:.6f}")
            logger.info(f"Max Absolute Error: {metrics['max_absolute_error']['max']:.6e}")
            
        except Exception as e:
            validation_result['status'] = 'error'
            validation_result['error'] = str(e)
            logger.error(f"Validation error: {str(e)}")
        
        return validation_result
    
    def validate_all_models(self, tflite_version='default', num_test_samples=10):
        """
        Validate all converted TFLite models.
        
        Args:
            tflite_version: Which TFLite version to validate ('default', 'float16', 'dynamic_range')
            num_test_samples: Number of test samples per model
        """
        logger.info("\n" + "="*80)
        logger.info(f"Starting validation of all TFLite models (version: {tflite_version})")
        logger.info("="*80 + "\n")
        
        # Find all model directories
        model_dirs = [d for d in self.tflite_models_dir.iterdir() if d.is_dir() and d.name != '_temp']
        
        if not model_dirs:
            logger.warning(f"No TFLite model directories found in {self.tflite_models_dir}")
            return
        
        logger.info(f"Found {len(model_dirs)} models to validate\n")
        
        passed = 0
        failed = 0
        errors = 0
        
        for idx, model_dir in enumerate(model_dirs, 1):
            model_name = model_dir.name
            
            # Find corresponding ONNX model
            onnx_path = self.onnx_models_dir / model_name / f"{model_name}.onnx"
            
            # Find TFLite model based on version
            if tflite_version == 'default':
                tflite_filename = f"{model_name}.tflite"
            elif tflite_version == 'float16':
                tflite_filename = f"{model_name}_float16.tflite"
            elif tflite_version == 'dynamic_range':
                tflite_filename = f"{model_name}_dynamic.tflite"
            else:
                tflite_filename = f"{model_name}.tflite"
            
            tflite_path = model_dir / tflite_filename
            
            # Check if both files exist
            if not onnx_path.exists():
                logger.warning(f"ONNX model not found: {onnx_path}")
                continue
            
            if not tflite_path.exists():
                logger.warning(f"TFLite model not found: {tflite_path}")
                continue
            
            logger.info(f"\n[{idx}/{len(model_dirs)}] Validating: {model_name}")
            
            # Validate model
            result = self.validate_model(model_name, onnx_path, tflite_path, num_test_samples)
            self.validation_results['models'][model_name] = result
            
            if result['status'] in ['passed', 'passed_with_minor_differences']:
                passed += 1
            elif result['status'] == 'failed':
                failed += 1
            else:
                errors += 1
        
        # Generate summary
        total = len(model_dirs)
        self.validation_results['summary'] = {
            'total_models': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'tflite_version_tested': tflite_version,
            'num_test_samples_per_model': num_test_samples
        }
        
        # Save validation report
        report_path = self.tflite_models_dir / f'validation_report_{tflite_version}.json'
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total models: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Errors: {errors}")
        logger.info(f"\nDetailed validation report saved to: {report_path}")
        logger.info("="*80 + "\n")
        
        return self.validation_results


def main():
    """Main execution function."""
    
    # Configuration
    ONNX_MODELS_DIR = r"D:\Base-dir\onnx_models"
    TFLITE_MODELS_DIR = r"D:\Base-dir\tflite_models"
    TFLITE_VERSION = 'default'  # Options: 'default', 'float16', 'dynamic_range'
    NUM_TEST_SAMPLES = 10
    
    print("\n" + "="*80)
    print("TFLite Model Accuracy Validation Tool")
    print("="*80)
    print(f"\nONNX models directory: {ONNX_MODELS_DIR}")
    print(f"TFLite models directory: {TFLITE_MODELS_DIR}")
    print(f"TFLite version to validate: {TFLITE_VERSION}")
    print(f"Number of test samples: {NUM_TEST_SAMPLES}")
    print("\n" + "="*80 + "\n")
    
    # Create validator instance
    validator = ModelAccuracyValidator(ONNX_MODELS_DIR, TFLITE_MODELS_DIR)
    
    # Validate all models
    results = validator.validate_all_models(
        tflite_version=TFLITE_VERSION,
        num_test_samples=NUM_TEST_SAMPLES
    )
    
    # Print summary
    if results:
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)
        print(f"\nCheck the log file 'tflite_accuracy_validation.log' for detailed information")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
