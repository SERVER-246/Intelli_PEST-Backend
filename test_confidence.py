"""Test model confidence on real images."""
import torch
import sys
from pathlib import Path

sys.path.insert(0, '.')
from inference_server.engine.pytorch_inference import PyTorchInference

# Load model - use the original high-accuracy model
engine = PyTorchInference(
    model_path=Path('D:/KnowledgeDistillation/student_model_final.pth'),
    device='cpu'
)

print(f"Model classes: {engine.class_names}")
print(f"Num classes: {engine.num_classes}")
print()

# Test with images from each class
dataset_dir = Path('D:/IMAGE DATASET')
classes_to_test = ['Healthy', 'army worm', 'Top borer', 'termite']

for cls in classes_to_test:
    cls_dir = dataset_dir / cls
    if not cls_dir.exists():
        continue
    
    images = list(cls_dir.glob('*.jpg'))[:2]  # Test 2 images per class
    
    for img_path in images:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        
        result = engine.predict(img_bytes)
        pred_class = result['class_name']
        confidence = result['confidence'] * 100
        
        correct = "✓" if pred_class == cls else "✗"
        print(f"{correct} {cls:20} -> {pred_class:20} ({confidence:.1f}%)")

print("\nDone!")
