"""
Production TFLite Converter
===========================

Converts ONNX models produced by `onnx_converter.py` into TFLite format.
Strategy:
 1. Try ONNX -> TensorFlow SavedModel via `onnx-tf` (preferred when available).
 2. If not available, try ONNX -> Keras via `onnx2keras` then save Keras -> SavedModel.
 3. If both fail, delegate to `fallback_tflite_converter.FallbackTFLiteConverter`.

Generates three variants per model when possible: `default` (float32), `float16`, and `dynamic_range`.
Saves outputs to `D:/Base-dir/tflite_models/<model>/` and writes a `tflite_conversion_report.json`.
"""

import os
import json
import shutil
import logging
import datetime
from pathlib import Path
import traceback
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
import onnx

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger('ProductionTFLiteConverter')

# Configuration (imported from onnx_converter.Config)
class Config:
    """Configuration matching onnx_converter.Config"""
    BASE_DIR = Path(r"D:\Base-dir")
    ONNX_OUTPUT_DIR = BASE_DIR / "onnx_models"
    MODELS_TO_CONVERT = [
        'alexnet', 'resnet50', 'inception_v3', 'mobilenet_v2', 'efficientnet_b0',
        'darknet53', 'yolo11n-cls', 'ensemble_attention', 'ensemble_concat',
        'ensemble_cross', 'super_ensemble'
    ]

# Local imports from this package
try:
    from .fallback_tflite_converter import FallbackTFLiteConverter
except Exception:
    from fallback_tflite_converter import FallbackTFLiteConverter

# Optional converters
try:
    from onnx_tf.backend import prepare as onnx_tf_prepare
    ONNX_TF_AVAILABLE = True
except Exception as e:
    ONNX_TF_AVAILABLE = False
    logger.info(f"onnx-tf not available: {str(e)[:100]}")

try:
    from onnx2keras import onnx_to_keras
    ONNX2KERAS_AVAILABLE = True
    logger.info("onnx2keras is available and will be used for conversion")
except Exception as e:
    ONNX2KERAS_AVAILABLE = False
    logger.warning(f"onnx2keras not available: {e}")


def _apply_quantization_settings(converter: tf.lite.TFLiteConverter, quant: str):
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # Some TF versions expose experimental flags; guard access
    try:
        converter._experimental_lower_tensor_list_ops = False
    except Exception:
        pass

    if quant == 'none':
        converter.optimizations = []
    elif quant == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        try:
            converter.target_spec.supported_types = [tf.float16]
        except Exception:
            pass


def convert_saved_model_to_tflite(saved_model_dir: Path, out_path: Path, quant: str = 'none') -> dict:
    """Convert a SavedModel directory to a TFLite file with optional quantization."""
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        _apply_quantization_settings(converter, quant)
        # Allow custom ops in case model uses them; they may fail on runtime though
        try:
            converter.allow_custom_ops = True
        except Exception:
            pass

        tflite_model = converter.convert()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(tflite_model)

        size_mb = len(tflite_model) / (1024 * 1024)
        return {'status': 'success', 'path': str(out_path), 'size_mb': round(size_mb, 2)}
    except Exception as e:
        logger.exception(f"TFLite conversion failed: {e}")
        return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}


def convert_onnx_to_saved_model_onnx_tf(onnx_path: Path, saved_model_dir: Path) -> dict:
    """Convert ONNX -> SavedModel using onnx-tf if available."""
    if not ONNX_TF_AVAILABLE:
        return {'status': 'skipped', 'reason': 'onnx-tf not available'}

    try:
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = onnx_tf_prepare(onnx_model)
        if saved_model_dir.exists():
            shutil.rmtree(saved_model_dir)
        tf_rep.export_graph(str(saved_model_dir))
        return {'status': 'success', 'saved_model_dir': str(saved_model_dir)}
    except Exception as e:
        logger.exception(f"onnx-tf conversion failed: {e}")
        return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}


def convert_onnx_to_saved_model_onnx2keras(onnx_path: Path, saved_model_dir: Path) -> dict:
    """Convert ONNX -> Keras -> SavedModel using onnx2keras if available."""
    if not ONNX2KERAS_AVAILABLE:
        return {'status': 'skipped', 'reason': 'onnx2keras not available'}

    try:
        onnx_model = onnx.load(str(onnx_path))
        # onnx2keras requires a list of input names; pick first
        input_name = onnx_model.graph.input[0].name
        
        k_model = onnx_to_keras(onnx_model, [input_name], change_ordering=True)
        # Save Keras model as SavedModel
        if saved_model_dir.exists():
            shutil.rmtree(saved_model_dir)
        k_model.save(str(saved_model_dir), save_format='tf')
        return {'status': 'success', 'saved_model_dir': str(saved_model_dir)}
    except Exception as e:
        logger.exception(f"onnx2keras conversion failed: {e}")
        return {'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc()}


def convert_model_production(model_id: str, onnx_dir: Path, tflite_dir: Path) -> dict:
    """Convert a single ONNX model to multiple TFLite variants.
    
    Workflow:
      - If ONNX model exists, try onnx-tf -> SavedModel -> TFLite.
      - Else try onnx2keras -> SavedModel -> TFLite.
      - If both attempts fail, call fallback converter to try additional methods.
    """
    report = {'model': model_id, 'timestamp': datetime.datetime.now().isoformat()}
    
    onnx_path = onnx_dir / model_id / f"{model_id}.onnx"
    if not onnx_path.exists():
        report.update({'status': 'skipped', 'reason': 'onnx not found', 'onnx_path': str(onnx_path)})
        logger.warning(f"ONNX not found for {model_id}: {onnx_path}")
        return report
    
    report['onnx_path'] = str(onnx_path)
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    report['onnx_size_mb'] = round(onnx_size_mb, 2)
    
    saved_model_temp = tflite_dir / '_temp' / f"{model_id}_saved_model"
    
    # Try onnx2keras first (more reliable with TF 2.15)
    method_result = convert_onnx_to_saved_model_onnx2keras(onnx_path, saved_model_temp)
    
    if method_result.get('status') != 'success':
        # Try onnx-tf if onnx2keras failed
        method_result = convert_onnx_to_saved_model_onnx_tf(onnx_path, saved_model_temp)
    
    if method_result.get('status') != 'success':
        # Delegate to fallback converter
        try:
            logger.info(f"Delegating {model_id} to fallback converter")
            fallback = FallbackTFLiteConverter(onnx_dir=onnx_dir, tflite_dir=tflite_dir, quantization=FallbackTFLiteConverter.QUANTIZATION_DYNAMIC)
            fallback_res = fallback.convert_model(model_id)
            report.update({'status': fallback_res.get('status', 'failed'), 'fallback': fallback_res})
            return report
        except Exception as e:
            logger.exception(f"Fallback converter failed for {model_id}: {e}")
            report.update({'status': 'failed', 'error': str(e)})
            return report
    
    # At this point we have saved_model_temp ready
    results = {}
    variants = {
        'default': 'none',
        'float16': 'float16',
        'dynamic_range': 'dynamic'
    }
    
    output_dir = tflite_dir / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ver, q in variants.items():
        tflite_name = f"{model_id}.tflite" if ver == 'default' else f"{model_id}_{ver}.tflite"
        tflite_path = output_dir / tflite_name
        conv = convert_saved_model_to_tflite(saved_model_temp, tflite_path, quant=q)
        results[ver] = conv
    
    # Save metadata
    try:
        meta = {
            'model': model_id,
            'onnx_path': str(onnx_path),
            'onnx_size_mb': report['onnx_size_mb'],
            'variants': results,
            'timestamp': datetime.datetime.now().isoformat()
        }
        with open(output_dir / 'tflite_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception:
        logger.exception("Failed to write metadata")
    
    # Cleanup temp saved model
    try:
        if saved_model_temp.exists():
            shutil.rmtree(saved_model_temp)
    except Exception:
        logger.exception("Failed to cleanup temp saved model")
    
    report.update({'status': 'converted', 'variants': results})
    logger.info(f"Converted {model_id} -> TFLite (variants: {', '.join(results.keys())})")
    return report


def main():
    ONNX_DIR = Config.ONNX_OUTPUT_DIR
    TFLITE_DIR = Path(r"D:\Base-dir\tflite_models")
    
    TFLITE_DIR.mkdir(parents=True, exist_ok=True)
    
    models = Config.MODELS_TO_CONVERT
    logger.info(f"Starting production TFLite conversion for {len(models)} models")
    
    overall = {'timestamp': datetime.datetime.now().isoformat(), 'models': {}}
    
    for mid in models:
        try:
            res = convert_model_production(mid, ONNX_DIR, TFLITE_DIR)
            overall['models'][mid] = res
        except Exception as e:
            logger.exception(f"Conversion failed for {mid}: {e}")
            overall['models'][mid] = {'status': 'error', 'error': str(e)}
    
    # Save overall report
    report_path = TFLITE_DIR / 'tflite_conversion_report.json'
    with open(report_path, 'w') as f:
        json.dump(overall, f, indent=2)
    
    logger.info(f"Production conversion complete. Report: {report_path}")


if __name__ == '__main__':
    main()
