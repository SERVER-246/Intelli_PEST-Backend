"""
Response Postprocessing
=======================
Format inference results for API responses.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:12]}"


def format_prediction(
    prediction_result: Dict[str, Any],
    request_id: Optional[str] = None,
    validation_result: Optional[Dict[str, Any]] = None,
    include_all_probabilities: bool = True,
) -> Dict[str, Any]:
    """
    Format a single prediction result for API response.
    
    Args:
        prediction_result: Raw prediction from inference engine
        request_id: Request identifier
        validation_result: Image validation result
        include_all_probabilities: Include all class probabilities
        
    Returns:
        Formatted API response dictionary
    """
    request_id = request_id or generate_request_id()
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Handle error case
    if not prediction_result.get("success", True):
        return {
            "status": "error",
            "request_id": request_id,
            "timestamp": timestamp,
            "error": {
                "code": "INFERENCE_ERROR",
                "message": prediction_result.get("error", "Unknown error during inference"),
            },
        }
    
    # Build successful response
    response = {
        "status": "success",
        "request_id": request_id,
        "timestamp": timestamp,
        "prediction": {
            "class": prediction_result.get("class_name", ""),
            "class_id": prediction_result.get("predicted_class", -1),
            "confidence": round(prediction_result.get("confidence", 0.0), 4),
        },
    }
    
    # Add all probabilities if requested
    if include_all_probabilities and "probabilities" in prediction_result:
        probs = prediction_result["probabilities"]
        if isinstance(probs, dict):
            response["prediction"]["all_probabilities"] = {
                k: round(v, 4) for k, v in probs.items()
            }
        elif hasattr(probs, '__iter__'):
            response["prediction"]["all_probabilities"] = [
                round(float(p), 4) for p in probs
            ]
    
    # Add inference metadata
    response["inference"] = {
        "model_format": prediction_result.get("model_format", "unknown"),
        "device": prediction_result.get("device", "unknown"),
        "time_ms": round(prediction_result.get("inference_time_ms", 0.0), 2),
    }
    
    # Add validation info if provided
    if validation_result:
        response["validation"] = {
            "passed": validation_result.get("valid", True),
            "relevance_score": round(validation_result.get("relevance_score", 1.0), 2),
            "quality_score": round(validation_result.get("quality_score", 1.0), 2),
        }
    
    return response


def format_batch_predictions(
    predictions: List[Dict[str, Any]],
    request_id: Optional[str] = None,
    validation_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Format batch prediction results for API response.
    
    Args:
        predictions: List of prediction results
        request_id: Request identifier
        validation_results: List of validation results
        
    Returns:
        Formatted API response dictionary
    """
    request_id = request_id or generate_request_id()
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Process each prediction
    results = []
    successful = 0
    failed = 0
    total_time = 0.0
    
    for i, pred in enumerate(predictions):
        validation = validation_results[i] if validation_results and i < len(validation_results) else None
        
        if pred.get("success", True):
            successful += 1
            results.append({
                "index": i,
                "status": "success",
                "prediction": {
                    "class": pred.get("class_name", ""),
                    "class_id": pred.get("predicted_class", -1),
                    "confidence": round(pred.get("confidence", 0.0), 4),
                },
                "inference_time_ms": round(pred.get("inference_time_ms", 0.0), 2),
            })
            total_time += pred.get("inference_time_ms", 0.0)
        else:
            failed += 1
            results.append({
                "index": i,
                "status": "error",
                "error": pred.get("error", "Unknown error"),
            })
    
    response = {
        "status": "success" if failed == 0 else "partial" if successful > 0 else "error",
        "request_id": request_id,
        "timestamp": timestamp,
        "summary": {
            "total": len(predictions),
            "successful": successful,
            "failed": failed,
            "total_time_ms": round(total_time, 2),
            "avg_time_ms": round(total_time / successful, 2) if successful > 0 else 0,
        },
        "results": results,
    }
    
    return response


def format_error_response(
    error_code: str,
    error_message: str,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    suggestion: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format an error response.
    
    Args:
        error_code: Error code string
        error_message: Human-readable error message
        request_id: Request identifier
        details: Additional error details
        suggestion: Suggestion for fixing the error
        
    Returns:
        Formatted error response
    """
    request_id = request_id or generate_request_id()
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    response = {
        "status": "error",
        "request_id": request_id,
        "timestamp": timestamp,
        "error": {
            "code": error_code,
            "message": error_message,
        },
    }
    
    if details:
        response["error"]["details"] = details
    
    if suggestion:
        response["suggestion"] = suggestion
    
    return response


def format_rejected_response(
    reason: str,
    request_id: Optional[str] = None,
    relevance_score: float = 0.0,
    detected_category: Optional[str] = None,
    suggestion: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format a rejection response for invalid/irrelevant images.
    
    Args:
        reason: Rejection reason
        request_id: Request identifier
        relevance_score: Content relevance score
        detected_category: Detected image category
        suggestion: Suggestion for user
        
    Returns:
        Formatted rejection response
    """
    request_id = request_id or generate_request_id()
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    return {
        "status": "rejected",
        "request_id": request_id,
        "timestamp": timestamp,
        "error": {
            "code": "CONTENT_IRRELEVANT",
            "message": reason,
            "details": {
                "relevance_score": round(relevance_score, 2),
                "threshold": 0.50,
                "detected_category": detected_category,
            },
        },
        "suggestion": suggestion or "Please upload a clear image of sugarcane plant or suspected pest damage",
    }


def format_health_response(
    healthy: bool = True,
    model_loaded: bool = False,
    model_info: Optional[Dict[str, Any]] = None,
    version: str = "1.0.0",
) -> Dict[str, Any]:
    """
    Format health check response.
    
    Args:
        healthy: Whether service is healthy
        model_loaded: Whether model is loaded
        model_info: Model information
        version: API version
        
    Returns:
        Health check response
    """
    return {
        "status": "healthy" if healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": version,
        "model": {
            "loaded": model_loaded,
            "info": model_info,
        },
    }
