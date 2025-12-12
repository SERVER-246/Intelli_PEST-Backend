"""Quick test of ONNX sanitizer"""
from onnx_sanitizer import ONNXSanitizer
from pathlib import Path
import onnx

print("Testing sanitizer on mobilenet_v2...")
onnx_path = Path(r'D:\Base-dir\onnx_models\mobilenet_v2\mobilenet_v2.onnx')

if not onnx_path.exists():
    print(f"ERROR: ONNX file not found: {onnx_path}")
    exit(1)

model = onnx.load(str(onnx_path))
sanitizer = ONNXSanitizer()
sanitized = sanitizer.sanitize_model(model)

print(f'✓ Sanitized {len(sanitizer.name_mapping)} names')
print(f'\nSample mappings (first 10):')
for i, (old, new) in enumerate(list(sanitizer.name_mapping.items())[:10]):
    print(f'  {old[:60]:60} -> {new[:60]}')

print(f'\n✓ Sanitization successful!')
