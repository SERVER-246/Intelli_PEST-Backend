"""
FastAPI Application Factory
===========================
Create and configure the FastAPI inference server.
"""

import logging
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import router, admin_router
from .app_management import app_router
from .dependencies import (
    set_inference_engine,
    set_validation_pipeline,
    set_api_key_manager,
    set_settings,
)

logger = logging.getLogger(__name__)


def create_app(
    config: Optional[dict] = None,
    model_path: Optional[str] = None,
    model_format: Optional[str] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config: Optional configuration dictionary
        model_path: Path to the model file
        model_format: Model format (pytorch, onnx, tflite)
        
    Returns:
        Configured FastAPI application
    """
    config = config or {}
    
    # Initialize components before app starts
    api_key_manager = None
    validation_pipeline = None
    inference_engine = None
    
    # Try to load security components
    try:
        from ..security import APIKeyManager, SecurityHeaders
        from .dependencies import _api_key_manager
        
        # Use existing manager if already set, otherwise create new one
        if _api_key_manager is not None:
            api_key_manager = _api_key_manager
            logger.info("Using pre-configured API key manager")
        else:
            api_key_manager = APIKeyManager()
            
            # Generate default key if none exist
            if api_key_manager.get_stats().get("total_keys", 0) == 0:
                default_key = api_key_manager.generate_key(
                    name="default",
                    tier="admin",
                    description="Default admin key",
                )
                logger.info(f"Generated default API key: {default_key['key']}")
            
            set_api_key_manager(api_key_manager)
        
    except Exception as e:
        logger.warning(f"Security components not available: {e}")
    
    # Try to load validation pipeline
    try:
        from ..filters import ValidationPipeline
        from ..filters.file_validator import FileValidator
        from ..filters.image_validator import ImageValidator
        from ..filters.content_filter import ContentFilter
        
        # Create more permissive validators for agricultural images
        # - Allow larger files (DSC camera images can be 15MB+)
        # - Allow higher resolution (up to 8192px for high-res cameras)
        # - Skip malicious content check (too many false positives on binary image data)
        file_validator = FileValidator(
            min_size=1024,                    # 1 KB minimum
            max_size=20 * 1024 * 1024,        # 20 MB maximum (was 10MB)
        )
        
        image_validator = ImageValidator(
            min_dimension=64,
            max_dimension=8192,               # Allow up to 8K images (was 4096)
            min_aspect_ratio=0.1,             # More permissive aspect ratio
            max_aspect_ratio=10.0,
        )
        
        # Configure content filter with relaxed settings but human detection
        # - Lower vegetation threshold (pest damage images may not be very green)
        # - Keep face/skin detection to reject obviously wrong images
        content_filter = ContentFilter(
            relevance_threshold=0.25,         # Lower threshold for relevance
            min_vegetation_ratio=0.05,        # Very permissive (pest damage may be brown)
            min_natural_score=0.20,           # Allow more types of natural images
        )
        
        validation_pipeline = ValidationPipeline(
            file_validator=file_validator,
            image_validator=image_validator,
            content_filter=content_filter,
            skip_content_filter=False,        # Enable content filter to reject non-plant images
        )
        set_validation_pipeline(validation_pipeline)
        logger.info("Image validation pipeline loaded (with human detection filter enabled)")
        
    except Exception as e:
        logger.warning(f"Validation pipeline not available: {e}")
        
    except Exception as e:
        logger.warning(f"Validation pipeline not available: {e}")
    
    # Try to load inference engine
    if model_path:
        try:
            from ..engine.pytorch_inference import PyTorchInference
            from pathlib import Path
            
            # Load PyTorch model directly (handles state dict reconstruction)
            inference_engine = PyTorchInference(
                model_path=Path(model_path),
                device="cuda" if model_format != "cpu" else "cpu",
            )
            set_inference_engine(inference_engine)
            logger.info(f"Inference engine loaded: {model_path}")
            
            # Initialize Phase 3 separately for router-level integration
            try:
                import sys
                # Add black_ops_training to path for Phase 3 imports
                black_ops_dir = Path(__file__).parent.parent.parent / "black_ops_training"
                logger.info(f"Looking for Phase 3 at: {black_ops_dir}")
                
                if black_ops_dir.exists():
                    if str(black_ops_dir) not in sys.path:
                        sys.path.insert(0, str(black_ops_dir))
                    
                    from phase3_enabled_config import enable_phase3, Phase3ProductionMode
                    phase3_manager = enable_phase3(Phase3ProductionMode.INFERENCE)
                    
                    # Store Phase 3 manager in app state for router access
                    from . import dependencies
                    dependencies._phase3_manager = phase3_manager
                    logger.info(f"Phase 3 initialized: manager operational = {phase3_manager.is_operational()}")
                else:
                    logger.warning(f"Phase 3 directory not found: {black_ops_dir}")
            except Exception as p3_err:
                import traceback
                logger.warning(f"Phase 3 initialization failed (continuing without): {p3_err}")
                traceback.print_exc()
                
        except Exception as e:
            logger.error(f"Failed to load inference engine: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("No model path provided, inference not available")
    
    # Create FastAPI app
    app = FastAPI(
        title="Sugarcane Pest Detection API",
        description="AI-powered sugarcane pest detection using deep learning models",
        version=config.get("version", "1.0.0"),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Store start time
    app.state.start_time = time.time()
    app.state.config = config
    
    # Add CORS middleware
    cors_origins = config.get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start_time) * 1000
        response.headers["X-Response-Time"] = f"{elapsed:.2f}ms"
        return response
    
    # Add security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal server error occurred",
                },
            },
        )
    
    # Include routers
    app.include_router(router)
    app.include_router(admin_router)
    app.include_router(app_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Sugarcane Pest Detection API",
            "version": config.get("version", "1.0.0"),
            "status": "running",
            "documentation": "/docs",
        }
    
    logger.info("FastAPI application created successfully")
    
    return app


def run_app(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    model_path: Optional[str] = None,
    model_format: Optional[str] = None,
):
    """
    Run the FastAPI application.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload
        model_path: Path to the model file
        model_format: Model format
    """
    import uvicorn
    
    # Create app with model
    app = create_app(model_path=model_path, model_format=model_format)
    
    logger.info(f"Starting FastAPI server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )


# Create default app instance only when run directly (not imported)
# This prevents the "No model path" warning when imported by run_server.py
app = None


def get_app():
    """Get or create the default app instance."""
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FastAPI Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--format", choices=["pytorch", "onnx", "tflite"],
                        help="Model format (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    run_app(
        host=args.host,
        port=args.port,
        reload=args.reload,
        model_path=args.model,
        model_format=args.format,
    )
