"""
FastAPI Dependencies
====================
Dependency injection for FastAPI routes.
"""

import logging
import time
from typing import Optional, Generator

from fastapi import Depends, HTTPException, Header, Query, Request
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# Globals (set during app initialization)
_inference_engine = None
_validation_pipeline = None
_api_key_manager = None
_settings = None


def set_inference_engine(engine):
    """Set the global inference engine."""
    global _inference_engine
    _inference_engine = engine


def set_validation_pipeline(pipeline):
    """Set the global validation pipeline."""
    global _validation_pipeline
    _validation_pipeline = pipeline


def set_api_key_manager(manager):
    """Set the global API key manager."""
    global _api_key_manager
    _api_key_manager = manager


def set_settings(settings):
    """Set global settings."""
    global _settings
    _settings = settings


# API Key Header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_inference_engine():
    """Get the inference engine dependency."""
    if _inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_NOT_LOADED",
                "message": "Inference model is not loaded",
            },
        )
    return _inference_engine


async def get_optional_inference_engine():
    """Get optional inference engine (may be None)."""
    return _inference_engine


async def get_validation_pipeline():
    """Get the validation pipeline (may be None)."""
    return _validation_pipeline


async def get_api_key_manager():
    """Get the API key manager (may be None)."""
    return _api_key_manager


async def verify_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    api_key_query: Optional[str] = Query(None, alias="api_key"),
) -> dict:
    """
    Verify API key from header or query parameter.
    
    Returns:
        Dictionary with tier and key_id
    """
    key = api_key or api_key_query
    
    if not key:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "MISSING_API_KEY",
                "message": "API key is required. Provide via X-API-Key header or api_key parameter.",
            },
        )
    
    if _api_key_manager is None:
        # No key manager, accept any key
        return {"tier": "free", "key_id": "unknown"}
    
    validation = _api_key_manager.validate_key(key)
    
    if not validation["valid"]:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "INVALID_API_KEY",
                "message": validation.get("error", "Invalid API key"),
            },
        )
    
    return {
        "tier": validation.get("tier", "free"),
        "key_id": validation.get("key_id", "unknown"),
    }


async def optional_api_key(
    api_key: Optional[str] = Depends(api_key_header),
    api_key_query: Optional[str] = Query(None, alias="api_key"),
) -> Optional[dict]:
    """
    Optional API key verification.
    
    Returns:
        Dictionary with tier info or None
    """
    key = api_key or api_key_query
    
    if not key:
        return None
    
    if _api_key_manager is None:
        return {"tier": "free", "key_id": "unknown"}
    
    validation = _api_key_manager.validate_key(key)
    
    if validation["valid"]:
        return {
            "tier": validation.get("tier", "free"),
            "key_id": validation.get("key_id", "unknown"),
        }
    
    return None


async def check_rate_limit(
    api_key: dict = Depends(verify_api_key),
    api_key_header_val: Optional[str] = Depends(api_key_header),
    api_key_query: Optional[str] = Query(None, alias="api_key"),
) -> dict:
    """
    Check rate limit for the API key.
    
    Returns:
        API key info if rate limit passed
    """
    key = api_key_header_val or api_key_query
    
    if key and _api_key_manager:
        allowed, info = _api_key_manager.check_rate_limit(key)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded. Please wait before making more requests.",
                    "retry_after_seconds": info.get("retry_after", 60),
                },
            )
        
        # Record the request
        _api_key_manager.record_request(key)
    
    return api_key


async def require_admin(api_key: dict = Depends(verify_api_key)) -> dict:
    """
    Require admin tier API key.
    
    Returns:
        API key info if admin
    """
    if api_key.get("tier") != "admin":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "FORBIDDEN",
                "message": "Admin access required",
            },
        )
    
    return api_key


class RequestContext:
    """Request context for tracking."""
    
    def __init__(self):
        self.request_id: str = f"req_{int(time.time() * 1000)}"
        self.start_time: float = time.time()
        self.api_key_info: Optional[dict] = None
    
    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


async def get_request_context(request: Request) -> RequestContext:
    """Get request context dependency."""
    ctx = RequestContext()
    ctx.request_id = request.headers.get("X-Request-ID", ctx.request_id)
    return ctx
