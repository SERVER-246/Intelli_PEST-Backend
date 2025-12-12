"""Test conversion with sanitized ONNX"""
from production_tflite_converter import convert_model_production
from pathlib import Path

ONNX_DIR = Path(r"D:\Base-dir\onnx_models")
TFLITE_DIR = Path(r"D:\Base-dir\tflite_models")

print("\n" + "="*70)
print("Testing conversion with sanitization + Clip fix")
print("="*70)
print("\nConverting: mobilenet_v2\n")

result = convert_model_production('mobilenet_v2', ONNX_DIR, TFLITE_DIR)

print("\n" + "="*70)
print("RESULT")
print("="*70)
print(f"Status: {result.get('status')}")
if result.get('status') == 'converted':
    print("\n✓ SUCCESS!")
    if 'variants' in result:
        for variant, info in result['variants'].items():
            if info.get('status') == 'success':
                print(f"  {variant}: {info.get('size_mb')} MB")
else:
    print(f"\nFAILED: {result.get('error', 'Unknown error')[:200]}")
