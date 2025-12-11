# Deployment Guide

## Overview

This guide explains how to deploy your models in production environments.

## Deployment Package Contents

Each deployed model includes:

```
model_name_deployment/
├── model.pth                  # PyTorch weights
├── model.onnx                 # ONNX format
├── model.tflite              # TensorFlow Lite formats
│   ├── _default.tflite       # Full precision
│   ├── _float16.tflite       # Float16 optimized
│   └── _dynamic.tflite       # Highly compressed
├── model_state_dict.pth      # State dict backup
├── metadata.json             # Model metadata
├── class_mapping.json        # Class names mapping
└── readme.txt                # Deployment notes
```

## PyTorch Deployment

### Load Model

```python
import torch
from pathlib import Path

# Load model
checkpoint = torch.load('model_deployment/model.pth', map_location='cpu')
model_state = checkpoint['model_state_dict']

# Reconstruct model (requires original architecture)
from src.training.base_training import create_classification_model
model, _ = create_classification_model('resnet50', num_classes=11)
model.load_state_dict(model_state)
model.eval()
```

### Inference

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load image
image = Image.open('pest.jpg').convert('RGB')

# Preprocess
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Inference
with torch.no_grad():
    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    confidence, class_idx = output.max(1)
    
print(f"Predicted: {class_names[class_idx.item()]}")
print(f"Confidence: {confidence.item():.2%}")
```

## ONNX Deployment

### Load and Infer

```python
import onnxruntime
import numpy as np
from PIL import Image

# Create session
sess = onnxruntime.InferenceSession('model_deployment/model.onnx')

# Load and preprocess image
image = Image.open('pest.jpg').convert('RGB')
image = image.resize((256, 256))
image_array = np.array(image).astype('float32') / 255.0
image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
image_array = np.expand_dims(image_array, 0)  # Add batch dimension

# Normalize
image_array = (image_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

# Inference
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: image_array})
prediction = output[0]

# Get class
class_idx = np.argmax(prediction[0])
confidence = prediction[0][class_idx]

print(f"Predicted: {class_names[class_idx]}")
print(f"Confidence: {confidence:.2%}")
```

## TensorFlow Lite Deployment

### Mobile (Android/iOS)

```java
// Android example
import org.tensorflow.lite.Interpreter;

Interpreter tflite = new Interpreter(modelFile);
float[][][][] input = new float[1][256][256][3];
float[][] output = new float[1][11];

tflite.run(input, output);
```

### Python Inference

```python
import tensorflow as tf
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input/output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input
input_data = np.random.rand(1, 256, 256, 3).astype('float32')

# Inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
```

## Batch Inference

### Multiple Images

```python
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = load_model('checkpoint.pth')
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])

# Process directory
results = []
image_dir = Path('images')
for img_path in image_dir.glob('*.jpg'):
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        confidence, class_idx = output.max(1)
    
    results.append({
        'image': img_path.name,
        'class': class_names[class_idx.item()],
        'confidence': confidence.item()
    })

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## Web Deployment (FastAPI)

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io

app = FastAPI()

# Load model once at startup
model = load_model('checkpoint.pth')
model.eval()
transform = get_transforms()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    
    confidence, class_idx = output.max(1)
    
    return {
        "class": class_names[class_idx.item()],
        "confidence": float(confidence.item()),
        "filename": file.filename
    }
```

## Performance Optimization

### Model Quantization

```python
# Already done during conversion, but can quantize PyTorch models:
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### GPU Inference

```python
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_tensor = input_tensor.to(device)

with torch.no_grad():
    output = model(input_tensor)
```

## Monitoring & Logging

### Log Predictions

```python
import logging

logging.basicConfig(
    filename='predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log each prediction
logger.info(f"Image: {filename} | Class: {class_name} | Conf: {confidence:.2%}")
```

### Track Performance

```python
from collections import defaultdict

metrics = defaultdict(list)

# During inference
metrics['inference_time'].append(elapsed_time)
metrics['confidence'].append(confidence)
metrics['class_distribution'][class_name] += 1

# Analyze
print(f"Avg Confidence: {np.mean(metrics['confidence']):.2%}")
print(f"Avg Inference Time: {np.mean(metrics['inference_time'])*1000:.2f}ms")
```

## Best Practices

1. **Always validate** before deploying
2. **Monitor performance** in production
3. **Log predictions** for audit trails
4. **Use appropriate format**:
   - PyTorch: Research, fine-tuning
   - ONNX: Cross-platform production
   - TFLite: Mobile/edge devices
5. **Implement fallback** for model failures
6. **Version your models** for reproducibility

## Troubleshooting

### Shape Mismatch
```python
# Debug input/output shapes
print(f"Model expects: {input_details}")
print(f"Input provided: {input_data.shape}")
```

### Low Confidence
- Try different model version
- Check image preprocessing
- Review training data distribution

### Slow Inference
- Use TFLite for speed
- Optimize batch size
- Consider model compression

## Next Steps

- Monitor model performance in production
- Collect user feedback
- Retrain with new data periodically
- Update deployment regularly
