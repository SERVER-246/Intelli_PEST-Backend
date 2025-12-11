"""
Model Comparison Tool
Generates detailed comparison between ONNX and TFLite models.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class ModelComparisonAnalyzer:
    """Analyze and compare ONNX and TFLite models."""
    
    def __init__(self, onnx_dir, tflite_dir):
        """Initialize analyzer."""
        self.onnx_dir = Path(onnx_dir)
        self.tflite_dir = Path(tflite_dir)
        
        # Load reports
        self.onnx_report = self.load_json(self.onnx_dir / 'conversion_report.json')
        self.tflite_report = self.load_json(self.tflite_dir / 'tflite_conversion_report.json')
        self.validation_report = self.load_json(self.tflite_dir / 'validation_report_default.json')
    
    def load_json(self, filepath):
        """Load JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            return {}
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return {}
    
    def generate_comparison_table(self):
        """Generate comprehensive comparison table."""
        print("\n" + "="*120)
        print("ONNX to TFLite Model Comparison Report")
        print("="*120 + "\n")
        
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Header
        header = f"{'Model':<20} {'ONNX':<12} {'Default':<12} {'Float16':<12} {'Dynamic':<12} {'Savings':<12} {'Accuracy':<12}"
        print(header)
        print("-" * 120)
        
        total_onnx_size = 0
        total_default_size = 0
        total_float16_size = 0
        total_dynamic_size = 0
        
        # Get model data
        models_data = []
        
        if self.onnx_report and self.tflite_report:
            for model_name in self.onnx_report.get('models', {}).keys():
                onnx_size = self.onnx_report['models'][model_name].get('file_size_mb', 0)
                
                tflite_model = self.tflite_report.get('models', {}).get(model_name, {})
                versions = tflite_model.get('versions', {})
                
                default_size = versions.get('default', {}).get('size_mb', 0)
                float16_size = versions.get('float16', {}).get('size_mb', 0)
                dynamic_size = versions.get('dynamic_range', {}).get('size_mb', 0)
                
                # Calculate size reduction
                if onnx_size > 0:
                    best_savings = max(
                        (onnx_size - dynamic_size) / onnx_size * 100 if dynamic_size else 0,
                        0
                    )
                else:
                    best_savings = 0
                
                # Get accuracy status
                validation = self.validation_report.get('models', {}).get(model_name, {})
                accuracy_status = validation.get('status', 'N/A')
                
                if accuracy_status == 'passed':
                    accuracy_icon = '✓ Preserved'
                elif accuracy_status == 'passed_with_minor_differences':
                    accuracy_icon = '~ Minor diff'
                else:
                    accuracy_icon = '✗ Check'
                
                models_data.append({
                    'name': model_name,
                    'onnx': onnx_size,
                    'default': default_size,
                    'float16': float16_size,
                    'dynamic': dynamic_size,
                    'savings': best_savings,
                    'accuracy': accuracy_icon
                })
                
                total_onnx_size += onnx_size
                total_default_size += default_size
                total_float16_size += float16_size
                total_dynamic_size += dynamic_size
                
                # Print row
                print(f"{model_name:<20} "
                      f"{onnx_size:>10.2f} MB "
                      f"{default_size:>10.2f} MB "
                      f"{float16_size:>10.2f} MB "
                      f"{dynamic_size:>10.2f} MB "
                      f"{best_savings:>9.1f}% "
                      f"{accuracy_icon:<12}")
        
        # Print totals
        print("-" * 120)
        total_savings = ((total_onnx_size - total_dynamic_size) / total_onnx_size * 100) if total_onnx_size > 0 else 0
        
        print(f"{'TOTAL':<20} "
              f"{total_onnx_size:>10.2f} MB "
              f"{total_default_size:>10.2f} MB "
              f"{total_float16_size:>10.2f} MB "
              f"{total_dynamic_size:>10.2f} MB "
              f"{total_savings:>9.1f}% ")
        
        print("\n" + "="*120)
        
        # Summary statistics
        print("\nSUMMARY STATISTICS")
        print("-" * 120)
        print(f"Total Models Converted: {len(models_data)}")
        print(f"\nOriginal ONNX Models:")
        print(f"  Total Size: {total_onnx_size:.2f} MB")
        print(f"\nTFLite Models:")
        print(f"  Default (Full Precision): {total_default_size:.2f} MB ({(total_default_size/total_onnx_size*100):.1f}% of original)")
        print(f"  Float16 Quantized:        {total_float16_size:.2f} MB ({(total_float16_size/total_onnx_size*100):.1f}% of original)")
        print(f"  Dynamic Range Quantized:  {total_dynamic_size:.2f} MB ({(total_dynamic_size/total_onnx_size*100):.1f}% of original)")
        print(f"\nTotal Space Saved (Dynamic): {total_onnx_size - total_dynamic_size:.2f} MB ({total_savings:.1f}%)")
        print(f"Combined Storage Required: {total_onnx_size + total_default_size + total_float16_size + total_dynamic_size:.2f} MB")
        
        # Accuracy statistics
        if self.validation_report:
            summary = self.validation_report.get('summary', {})
            print(f"\nAccuracy Validation:")
            print(f"  Passed: {summary.get('passed', 0)}/{summary.get('total_models', 0)} models")
            print(f"  Failed: {summary.get('failed', 0)} models")
            print(f"  Errors: {summary.get('errors', 0)} models")
        
        print("\n" + "="*120 + "\n")
        
        return models_data, {
            'total_onnx': total_onnx_size,
            'total_default': total_default_size,
            'total_float16': total_float16_size,
            'total_dynamic': total_dynamic_size,
            'total_savings': total_savings
        }
    
    def generate_detailed_accuracy_report(self):
        """Generate detailed accuracy comparison."""
        print("\n" + "="*120)
        print("DETAILED ACCURACY METRICS")
        print("="*120 + "\n")
        
        if not self.validation_report:
            print("Validation report not found. Run validation first.")
            return
        
        header = f"{'Model':<20} {'MAE':<15} {'Cosine Sim':<15} {'Max Error':<15} {'Status':<20}"
        print(header)
        print("-" * 120)
        
        for model_name, details in self.validation_report.get('models', {}).items():
            if 'summary' in details:
                summary = details['summary']
                mae = summary.get('mean_absolute_error', 0)
                cosine = summary.get('cosine_similarity', 0)
                max_error = summary.get('max_absolute_error', 0)
                status = details.get('status', 'unknown')
                
                print(f"{model_name:<20} "
                      f"{mae:<15.2e} "
                      f"{cosine:<15.6f} "
                      f"{max_error:<15.2e} "
                      f"{status:<20}")
        
        print("\n" + "="*120)
        print("\nMETRIC DEFINITIONS:")
        print("-" * 120)
        print("MAE (Mean Absolute Error):  Average difference between ONNX and TFLite outputs")
        print("                            - Excellent: < 1e-6")
        print("                            - Good:      < 1e-3")
        print("                            - Acceptable: < 1e-2")
        print("\nCosine Similarity:          Measures output vector similarity (1.0 = identical)")
        print("                            - Excellent: > 0.9999")
        print("                            - Good:      > 0.999")
        print("                            - Acceptable: > 0.99")
        print("\nMax Error:                  Largest single difference found")
        print("\n" + "="*120 + "\n")
    
    def save_text_report(self, output_path='model_comparison_report.txt'):
        """Save comparison report to text file."""
        import sys
        from io import StringIO
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Generate reports
        self.generate_comparison_table()
        self.generate_detailed_accuracy_report()
        
        # Get captured output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Save to file
        output_file = self.tflite_dir / output_path
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"\n✓ Comparison report saved to: {output_file}\n")
        
        # Print to console
        print(output)


def main():
    """Main execution."""
    
    ONNX_DIR = r"D:\Base-dir\onnx_models"
    TFLITE_DIR = r"D:\Base-dir\tflite_models"
    
    print("\n" + "="*120)
    print("Model Comparison and Analysis Tool")
    print("="*120 + "\n")
    
    # Create analyzer
    analyzer = ModelComparisonAnalyzer(ONNX_DIR, TFLITE_DIR)
    
    # Generate reports
    analyzer.save_text_report('model_comparison_report.txt')
    
    print("="*120)
    print("\nAnalysis complete!")
    print(f"Reports available in: {TFLITE_DIR}")
    print("="*120 + "\n")


if __name__ == "__main__":
    main()
