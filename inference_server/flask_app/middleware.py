"""
Flask Middleware
================
Authentication, rate limiting, and security middleware for Flask.
"""

import logging
import time
from functools import wraps
from typing import Callable, Optional

from flask import Flask, Request, Response, request, g, jsonify

logger = logging.getLogger(__name__)


def init_middleware(app: Flask, api_key_manager=None, security_headers=None):
    """
    Initialize all middleware for the Flask application.
    
    Args:
        app: Flask application instance
        api_key_manager: API key manager instance
        security_headers: Security headers manager instance
    """
    
    @app.before_request
    def before_request():
        """Handle pre-request processing."""
        g.request_start_time = time.time()
        g.request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
    
    @app.after_request
    def after_request(response: Response) -> Response:
        """Handle post-request processing."""
        # Add timing header
        if hasattr(g, "request_start_time"):
            elapsed = (time.time() - g.request_start_time) * 1000
            response.headers["X-Response-Time"] = f"{elapsed:.2f}ms"
        
        # Add request ID
        if hasattr(g, "request_id"):
            response.headers["X-Request-ID"] = g.request_id
        
        # Add security headers
        if security_headers:
            headers = security_headers.get_headers()
            for key, value in headers.items():
                response.headers[key] = value
        else:
            # Default security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Log request
        logger.info(
            f"{request.method} {request.path} - "
            f"Status: {response.status_code} - "
            f"Time: {response.headers.get('X-Response-Time', 'N/A')}"
        )
        
        return response
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        """Global exception handler."""
        logger.exception(f"Unhandled exception: {e}")
        return jsonify({
            "status": "error",
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred",
            }
        }), 500


def require_api_key(api_key_manager=None):
    """
    Decorator for routes requiring API key authentication.
    
    Args:
        api_key_manager: API key manager instance
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get API key from header or query parameter
            api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
            
            if not api_key:
                return jsonify({
                    "status": "error",
                    "error": {
                        "code": "MISSING_API_KEY",
                        "message": "API key is required. Provide via X-API-Key header or api_key parameter.",
                    }
                }), 401
            
            # Validate API key
            if api_key_manager:
                validation = api_key_manager.validate_key(api_key)
                if not validation["valid"]:
                    return jsonify({
                        "status": "error",
                        "error": {
                            "code": "INVALID_API_KEY",
                            "message": validation.get("error", "Invalid API key"),
                        }
                    }), 401
                
                # Store tier info in g
                g.api_tier = validation.get("tier", "free")
                g.api_key_id = validation.get("key_id", "unknown")
            else:
                g.api_tier = "free"
                g.api_key_id = "default"
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def rate_limit(api_key_manager=None, tier_limits: Optional[dict] = None):
    """
    Decorator for rate limiting routes.
    
    Args:
        api_key_manager: API key manager instance
        tier_limits: Custom tier limits override
        
    Returns:
        Decorator function
    """
    default_limits = {
        "free": {"requests_per_minute": 10, "requests_per_day": 100},
        "standard": {"requests_per_minute": 60, "requests_per_day": 1000},
        "premium": {"requests_per_minute": 300, "requests_per_day": 10000},
        "admin": {"requests_per_minute": 1000, "requests_per_day": 100000},
    }
    
    limits = tier_limits or default_limits
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if api_key_manager:
                api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
                
                if api_key:
                    allowed, info = api_key_manager.check_rate_limit(api_key)
                    
                    if not allowed:
                        return jsonify({
                            "status": "error",
                            "error": {
                                "code": "RATE_LIMIT_EXCEEDED",
                                "message": "Rate limit exceeded. Please wait before making more requests.",
                                "details": {
                                    "retry_after_seconds": info.get("retry_after", 60),
                                }
                            }
                        }), 429
                    
                    # Record the request
                    api_key_manager.record_request(api_key)
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def validate_content_type(*allowed_types):
    """
    Decorator to validate Content-Type header.
    
    Args:
        *allowed_types: Allowed content types
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            content_type = request.content_type or ""
            
            if not any(ct in content_type for ct in allowed_types):
                return jsonify({
                    "status": "error",
                    "error": {
                        "code": "INVALID_CONTENT_TYPE",
                        "message": f"Invalid Content-Type. Allowed: {', '.join(allowed_types)}",
                    }
                }), 415
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def validate_request_size(max_size: int = 10 * 1024 * 1024):  # 10MB default
    """
    Decorator to validate request size.
    
    Args:
        max_size: Maximum request size in bytes
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.content_length and request.content_length > max_size:
                return jsonify({
                    "status": "error",
                    "error": {
                        "code": "REQUEST_TOO_LARGE",
                        "message": f"Request size exceeds limit of {max_size // 1024 // 1024}MB",
                    }
                }), 413
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator
