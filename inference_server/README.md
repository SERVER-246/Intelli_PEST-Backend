# Sugarcane Pest Detection Inference Server

A production-grade inference server for sugarcane pest detection using deep learning models. Supports multiple model formats (PyTorch, ONNX, TFLite) and web frameworks (Flask, FastAPI).

## Features

- 🚀 **Dual Framework Support**: Flask and FastAPI implementations
- 🧠 **Multi-Format Models**: PyTorch (.pth), ONNX (.onnx), TFLite (.tflite)
- 🔐 **Security**: API key authentication, rate limiting, input sanitization
- 🔍 **Image Filtering**: 4-layer validation to reject junk images
- 📊 **11 Pest Classes**: Detect 10 sugarcane pests + healthy classification
- 🌐 **Ngrok Integration**: Easy public URL for mobile app testing
- 🐳 **Docker Ready**: Containerized deployment with GPU support

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Server

**With FastAPI (recommended):**
```bash
python scripts/start_fastapi.py --model path/to/student_model.pth
```

**With Flask:**
```bash
python scripts/start_flask.py --model path/to/student_model.pth
```

**With Ngrok (public URL):**
```bash
python scripts/start_ngrok.py --model path/to/student_model.pth
```

### 3. Test the API

```bash
python scripts/test_client.py --url http://localhost:8000 --image test_image.jpg
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/classes` | GET | List pest classes |
| `/api/v1/models` | GET | List available models |
| `/api/v1/predict` | POST | Single image prediction |
| `/api/v1/predict/base64` | POST | Predict from base64 image |
| `/api/v1/predict/batch` | POST | Batch prediction |

## Usage Examples

### Single Prediction (File Upload)

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: your-api-key" \
  -F "image=@sugarcane_leaf.jpg"
```

### Single Prediction (Base64)

```bash
curl -X POST http://localhost:8000/api/v1/predict/base64 \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"image_data": "base64_encoded_image_here"}'
```

### Response Format

```json
{
  "status": "success",
  "request_id": "req_abc123",
  "timestamp": "2024-01-15T10:30:00Z",
  "prediction": {
    "class": "army worm",
    "class_id": 1,
    "confidence": 0.9523
  },
  "inference": {
    "model_format": "pytorch",
    "device": "cuda",
    "time_ms": 45.23
  }
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | - | Path to model file |
| `MODEL_FORMAT` | auto | Model format (pytorch/onnx/tflite) |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `LOG_LEVEL` | INFO | Logging level |
| `NGROK_AUTHTOKEN` | - | Ngrok authentication token |

### API Key Tiers

| Tier | Rate Limit | Features |
|------|-----------|----------|
| free | 10 req/min | Basic inference |
| standard | 60 req/min | + Batch processing |
| premium | 300 req/min | + Priority support |
| admin | Unlimited | Full access |

## Docker Deployment

### Build Images

```bash
cd docker
docker-compose build
```

### Run Services

**FastAPI Server:**
```bash
docker-compose up fastapi-server
```

**With Ngrok:**
```bash
NGROK_AUTHTOKEN=your_token docker-compose --profile ngrok up
```

**GPU Version:**
```bash
docker-compose --profile gpu up
```

## Project Structure

```
inference_server/
├── config/              # Configuration files
│   ├── settings.py      # App settings
│   ├── model_config.yaml
│   └── security_config.yaml
├── security/            # Security components
│   ├── api_keys.py      # API key management
│   ├── audit.py         # Audit logging
│   ├── headers.py       # Security headers
│   └── sanitizer.py     # Input sanitization
├── filters/             # Image validation
│   ├── file_validator.py
│   ├── image_validator.py
│   ├── content_filter.py
│   ├── ood_detector.py
│   └── pipeline.py
├── engine/              # Inference engine
│   ├── model_loader.py
│   ├── pytorch_inference.py
│   ├── onnx_inference.py
│   ├── tflite_inference.py
│   ├── model_registry.py
│   └── inference.py
├── utils/               # Utilities
│   ├── preprocessing.py
│   ├── postprocessing.py
│   └── logger.py
├── flask_app/           # Flask application
│   ├── app.py
│   ├── routes.py
│   └── middleware.py
├── fastapi_app/         # FastAPI application
│   ├── main.py
│   ├── routers.py
│   ├── schemas.py
│   └── dependencies.py
├── docker/              # Docker files
│   ├── Dockerfile.flask
│   ├── Dockerfile.fastapi
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
└── scripts/             # Utility scripts
    ├── start_flask.py
    ├── start_fastapi.py
    ├── start_ngrok.py
    ├── generate_api_key.py
    └── test_client.py
```

## Pest Classes

1. **Healthy** - No pest damage detected
2. **Army worm** - Leaf damage from army worm infestation
3. **Internode borer** - Damage to sugarcane internodes
4. **Mealy bug** - White cottony masses on stems
5. **Pink borer** - Pink larvae boring into stems
6. **Porcupine damage** - Physical damage from porcupines
7. **Rat damage** - Gnaw marks and stem damage
8. **Root borer** - Root system damage
9. **Stalk borer** - Internal stem tunneling
10. **Termite** - Underground pest damage
11. **Top borer** - Damage to growing point

## Android Integration

To use with your Android app, follow these steps:

1. Start the server with ngrok:
   ```bash
   python scripts/start_ngrok.py --model student_model.pth
   ```

2. Copy the public URL displayed (e.g., `https://abc123.ngrok.io`)

3. Update your Android app's API endpoint to use this URL

4. Make prediction requests:
   ```kotlin
   val response = apiService.predict(
       apiKey = "your-api-key",
       image = imageFile.asRequestBody()
   )
   ```

## License

This project is part of the Intelli-PEST system for sugarcane pest detection.

## Support

For issues and questions, please open an issue on the repository.
