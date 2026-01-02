"""
Flask Routes
============
API routes for the Flask inference server.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from flask import Blueprint, request, jsonify, g, current_app

from .middleware import (
    require_api_key,
    rate_limit,
    validate_content_type,
    validate_request_size,
)

logger = logging.getLogger(__name__)

# Create blueprint
api = Blueprint("api", __name__, url_prefix="/api/v1")


def get_engine():
    """Get inference engine from app context."""
    return current_app.config.get("inference_engine")


def get_validation_pipeline():
    """Get validation pipeline from app context."""
    return current_app.config.get("validation_pipeline")


def get_api_key_manager():
    """Get API key manager from app context."""
    return current_app.config.get("api_key_manager")


# Health check - no auth required
@api.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    engine = get_engine()
    
    model_loaded = engine is not None
    model_info = None
    
    if engine:
        try:
            model_info = {
                "format": engine.model_format,
                "device": engine.device,
                "num_classes": engine.num_classes,
            }
        except Exception:
            pass
    
    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": current_app.config.get("version", "1.0.0"),
        "model": {
            "loaded": model_loaded,
            "info": model_info,
        },
    })


# Get available models info (public)
@api.route("/models", methods=["GET"])
def list_models():
    """List available models (public info only)."""
    from ..engine.model_registry import get_registry
    
    try:
        registry = get_registry()
        exposed_models = registry.get_exposed_models()
        
        return jsonify({
            "status": "success",
            "models": exposed_models,
        })
    except Exception as e:
        logger.exception("Error listing models")
        return jsonify({
            "status": "error",
            "error": {
                "code": "MODEL_LIST_ERROR",
                "message": str(e),
            }
        }), 500


# Single prediction endpoint
@api.route("/predict", methods=["POST"])
@require_api_key(api_key_manager=None)  # Will be set dynamically
@rate_limit(api_key_manager=None)
@validate_request_size(max_size=10 * 1024 * 1024)
def predict():
    """
    Single image prediction endpoint.
    
    Accepts:
    - multipart/form-data with 'image' file
    - application/json with base64 'image_data'
    """
    request_id = getattr(g, "request_id", f"req_{int(time.time() * 1000)}")
    start_time = time.time()
    
    engine = get_engine()
    pipeline = get_validation_pipeline()
    
    if engine is None:
        return jsonify({
            "status": "error",
            "request_id": request_id,
            "error": {
                "code": "MODEL_NOT_LOADED",
                "message": "Inference model is not loaded",
            }
        }), 503
    
    # Get image data
    image_bytes = None
    filename = "upload"
    
    if request.content_type and "multipart/form-data" in request.content_type:
        # File upload
        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "request_id": request_id,
                "error": {
                    "code": "NO_IMAGE_PROVIDED",
                    "message": "No image file provided. Use 'image' field in form data.",
                }
            }), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({
                "status": "error",
                "request_id": request_id,
                "error": {
                    "code": "EMPTY_FILENAME",
                    "message": "No file selected",
                }
            }), 400
        
        filename = file.filename
        image_bytes = file.read()
    
    elif request.content_type and "application/json" in request.content_type:
        # JSON with base64
        import base64
        
        data = request.get_json()
        if not data or "image_data" not in data:
            return jsonify({
                "status": "error",
                "request_id": request_id,
                "error": {
                    "code": "NO_IMAGE_DATA",
                    "message": "No image_data provided in JSON body",
                }
            }), 400
        
        try:
            image_bytes = base64.b64decode(data["image_data"])
        except Exception:
            return jsonify({
                "status": "error",
                "request_id": request_id,
                "error": {
                    "code": "INVALID_BASE64",
                    "message": "Invalid base64 image data",
                }
            }), 400
    
    else:
        return jsonify({
            "status": "error",
            "request_id": request_id,
            "error": {
                "code": "INVALID_CONTENT_TYPE",
                "message": "Use multipart/form-data or application/json",
            }
        }), 415
    
    # Validate image if pipeline is available
    if pipeline:
        validation_result = pipeline.validate(image_bytes, filename)
        
        if not validation_result.valid:
            return jsonify({
                "status": "rejected",
                "request_id": request_id,
                "error": {
                    "code": "IMAGE_VALIDATION_FAILED",
                    "message": validation_result.message,
                    "details": {
                        "failed_layer": validation_result.failed_layer,
                        "relevance_score": round(validation_result.relevance_score, 2),
                    }
                },
                "suggestion": "Please upload a clear image of sugarcane plant or suspected pest damage",
            }), 400
    
    # Run inference
    try:
        result = engine.predict(image_bytes)
        
        inference_time = (time.time() - start_time) * 1000
        
        response = {
            "status": "success",
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prediction": {
                "class": result.get("class_name", ""),
                "class_id": result.get("predicted_class", -1),
                "confidence": round(result.get("confidence", 0.0), 4),
            },
            "inference": {
                "model_format": engine.model_format,
                "device": engine.device,
                "time_ms": round(inference_time, 2),
            },
        }
        
        # Include all probabilities if requested
        if request.args.get("include_probabilities", "false").lower() == "true":
            probs = result.get("probabilities", {})
            if isinstance(probs, dict):
                response["prediction"]["all_probabilities"] = {
                    k: round(v, 4) for k, v in probs.items()
                }
        
        return jsonify(response)
    
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        return jsonify({
            "status": "error",
            "request_id": request_id,
            "error": {
                "code": "INFERENCE_ERROR",
                "message": str(e),
            }
        }), 500


# Batch prediction endpoint
@api.route("/predict/batch", methods=["POST"])
@require_api_key(api_key_manager=None)
@rate_limit(api_key_manager=None)
@validate_request_size(max_size=50 * 1024 * 1024)  # 50MB for batch
def predict_batch():
    """
    Batch image prediction endpoint.
    
    Accepts multipart/form-data with multiple 'images' files.
    """
    request_id = getattr(g, "request_id", f"req_{int(time.time() * 1000)}")
    start_time = time.time()
    
    engine = get_engine()
    pipeline = get_validation_pipeline()
    
    if engine is None:
        return jsonify({
            "status": "error",
            "request_id": request_id,
            "error": {
                "code": "MODEL_NOT_LOADED",
                "message": "Inference model is not loaded",
            }
        }), 503
    
    # Get images
    files = request.files.getlist("images")
    
    if not files:
        return jsonify({
            "status": "error",
            "request_id": request_id,
            "error": {
                "code": "NO_IMAGES_PROVIDED",
                "message": "No images provided. Use 'images' field in form data.",
            }
        }), 400
    
    # Limit batch size
    max_batch_size = current_app.config.get("max_batch_size", 10)
    if len(files) > max_batch_size:
        return jsonify({
            "status": "error",
            "request_id": request_id,
            "error": {
                "code": "BATCH_SIZE_EXCEEDED",
                "message": f"Maximum batch size is {max_batch_size} images",
            }
        }), 400
    
    results = []
    successful = 0
    failed = 0
    
    for i, file in enumerate(files):
        image_bytes = file.read()
        filename = file.filename or f"image_{i}"
        
        # Validate
        if pipeline:
            validation_result = pipeline.validate(image_bytes, filename)
            if not validation_result.valid:
                results.append({
                    "index": i,
                    "filename": filename,
                    "status": "rejected",
                    "error": validation_result.message,
                })
                failed += 1
                continue
        
        # Predict
        try:
            result = engine.predict(image_bytes)
            results.append({
                "index": i,
                "filename": filename,
                "status": "success",
                "prediction": {
                    "class": result.get("class_name", ""),
                    "class_id": result.get("predicted_class", -1),
                    "confidence": round(result.get("confidence", 0.0), 4),
                },
            })
            successful += 1
        except Exception as e:
            results.append({
                "index": i,
                "filename": filename,
                "status": "error",
                "error": str(e),
            })
            failed += 1
    
    total_time = (time.time() - start_time) * 1000
    
    return jsonify({
        "status": "success" if failed == 0 else "partial" if successful > 0 else "error",
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total": len(files),
            "successful": successful,
            "failed": failed,
            "total_time_ms": round(total_time, 2),
            "avg_time_ms": round(total_time / len(files), 2),
        },
        "results": results,
    })


# Classes info endpoint
@api.route("/classes", methods=["GET"])
def list_classes():
    """List pest classes."""
    engine = get_engine()
    
    if engine is None:
        # Return default classes
        classes = [
            "Healthy",
            "army worm",
            "Internode borer",
            "mealy bug",
            "Pink borer",
            "porcupine damage",
            "Rat damage",
            "root borer",
            "Stalk borer",
            "termite",
            "Top borer",
        ]
    else:
        classes = engine.class_names
    
    return jsonify({
        "status": "success",
        "num_classes": len(classes),
        "classes": [
            {"id": i, "name": name}
            for i, name in enumerate(classes)
        ],
    })


# Statistics endpoint (admin only)
@api.route("/stats", methods=["GET"])
@require_api_key(api_key_manager=None)
def get_stats():
    """Get API statistics (admin only)."""
    tier = getattr(g, "api_tier", "free")
    
    if tier != "admin":
        return jsonify({
            "status": "error",
            "error": {
                "code": "FORBIDDEN",
                "message": "Admin access required",
            }
        }), 403
    
    # Return statistics
    api_key_manager = get_api_key_manager()
    
    stats = {
        "uptime_seconds": time.time() - current_app.config.get("start_time", time.time()),
    }
    
    if api_key_manager:
        stats["api_keys"] = api_key_manager.get_stats()
    
    return jsonify({
        "status": "success",
        "stats": stats,
    })
