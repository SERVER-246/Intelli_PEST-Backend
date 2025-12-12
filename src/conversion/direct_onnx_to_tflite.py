"""
Direct ONNX to TFLite Converter using TensorFlow's ONNX support
================================================================
This converter uses tf2onnx's reverse path or direct TensorFlow ONNX loading
"""

import os
import sys
import json
import logging
from pathlib import Path
import subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DirectONNXToTFLite')

import tensorflow as tf
import onnx

def convert_onnx_via_tf2onnx_reverse(onnx_path: Path, tflite_path: Path) -> dict:
    """Try converting ONNX -> SavedModel -> TFLite using tf2onnx command line tool"""
    try:
        saved_model_temp = tflite_path.parent / f"_temp_{tflite_path.stem}_saved_model"
        saved_model_temp.mkdir(parents=True, exist_ok=True)
        
        # Use tf2onnx in reverse (onnx to saved_model) - experimental
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--onnx", str(onnx_path),
            "--output", str(saved_model_temp),
            "--opset", "13"
        ]
        
        logger.info(f"Attempting tf2onnx reverse conversion...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return {'status': 'failed', 'error': result.stderr}
        
        # Convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_temp))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        try:
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        except:
            pass
            
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        import shutil
        shutil.rmtree(saved_model_temp, ignore_errors=True)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        return {'status': 'success', 'path': str(tflite_path), 'size_mb': round(size_mb, 2)}
        
    except Exception as e:
        logger.exception(f"tf2onnx reverse conversion failed: {e}")
        return {'status': 'failed', 'error': str(e)}


def convert_remaining_models():
    """Convert the remaining 6 models that failed"""
    onnx_dir = Path(r"D:\Base-dir\onnx_models")
    tflite_dir = Path(r"D:\Base-dir\tflite_models")
    
    remaining_models = ['mobilenet_v2', 'darknet53']  # Start with these 2
    
    results = {}
    for model_id in remaining_models:
        logger.info(f"\n{'='*70}\nConverting: {model_id}\n{'='*70}")
        
        onnx_path = onnx_dir / model_id / f"{model_id}.onnx"
        if not onnx_path.exists():
            logger.warning(f"ONNX not found: {onnx_path}")
            results[model_id] = {'status': 'skipped', 'reason': 'onnx not found'}
            continue
        
        output_dir = tflite_dir / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        tflite_path = output_dir / f"{model_id}.tflite"
        
        result = convert_onnx_via_tf2onnx_reverse(onnx_path, tflite_path)
        results[model_id] = result
        
        logger.info(f"Result: {result.get('status')}")
    
    # Save results
    report_path = tflite_dir / 'direct_conversion_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nDirect conversion report saved to: {report_path}")
    return results


if __name__ == '__main__':
    convert_remaining_models()
