"""
Convert Remaining Models
========================
Converts the 6 blocked models using the sanitized ONNX approach.
"""

import sys
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from production_tflite_converter import convert_model_production, Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger('ConvertRemaining')

def main():
    ONNX_DIR = Path(r"D:\Base-dir\onnx_models")
    TFLITE_DIR = Path(r"D:\Base-dir\tflite_models")
    
    # Models that were previously blocked
    REMAINING_MODELS = [
        'mobilenet_v2',
        'darknet53',
        'ensemble_attention',
        'ensemble_concat',
        'ensemble_cross',
        'super_ensemble'
    ]
    
    print("\n" + "="*70)
    print("CONVERTING REMAINING MODELS WITH SANITIZATION")
    print("="*70)
    print(f"\nModels to convert: {len(REMAINING_MODELS)}")
    for model in REMAINING_MODELS:
        print(f"  - {model}")
    print()
    
    results = {}
    successful = 0
    failed = 0
    
    for idx, model_name in enumerate(REMAINING_MODELS, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(REMAINING_MODELS)}] Converting: {model_name}")
        print(f"{'='*70}")
        
        try:
            result = convert_model_production(model_name, ONNX_DIR, TFLITE_DIR)
            results[model_name] = result
            
            if result.get('status') in ['converted', 'success']:
                successful += 1
                print(f"✓ SUCCESS: {model_name}")
                if 'variants' in result:
                    for variant, info in result['variants'].items():
                        if info.get('status') == 'success':
                            print(f"  - {variant}: {info.get('size_mb', 'N/A')} MB")
            else:
                failed += 1
                print(f"✗ FAILED: {model_name}")
                if 'error' in result:
                    print(f"  Error: {result['error'][:200]}")
        
        except Exception as e:
            failed += 1
            print(f"✗ ERROR: {model_name}")
            print(f"  Exception: {str(e)[:200]}")
            results[model_name] = {'status': 'error', 'error': str(e)}
    
    # Summary
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"Total:      {len(REMAINING_MODELS)}")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"{'='*70}\n")
    
    # Save results
    import json
    report_path = TFLITE_DIR / 'remaining_models_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Report saved: {report_path}\n")
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
