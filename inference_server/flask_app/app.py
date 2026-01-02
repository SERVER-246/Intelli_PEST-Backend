"""
Flask Application Factory
=========================
Create and configure the Flask inference server.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from flask import Flask
from flask_cors import CORS

from .routes import api
from .middleware import init_middleware

logger = logging.getLogger(__name__)


def create_app(
    config: Optional[dict] = None,
    model_path: Optional[str] = None,
    model_format: Optional[str] = None,
) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Optional configuration dictionary
        model_path: Path to the model file
        model_format: Model format (pytorch, onnx, tflite)
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    app.config["start_time"] = time.time()
    
    # Load configuration
    if config:
        app.config.update(config)
    
    # Set defaults
    app.config.setdefault("version", "1.0.0")
    app.config.setdefault("max_batch_size", 10)
    app.config.setdefault("MAX_CONTENT_LENGTH", 50 * 1024 * 1024)  # 50MB
    
    # Initialize CORS
    cors_origins = app.config.get("cors_origins", ["*"])
    CORS(app, origins=cors_origins)
    
    # Initialize components
    api_key_manager = None
    security_headers = None
    validation_pipeline = None
    inference_engine = None
    
    # Try to load security components
    try:
        from ..security import APIKeyManager, SecurityHeaders
        
        api_key_manager = APIKeyManager()
        security_headers = SecurityHeaders()
        
        # Generate default key if none exist
        if api_key_manager.get_stats().get("total_keys", 0) == 0:
            default_key = api_key_manager.generate_key(
                name="default",
                tier="admin",
                description="Default admin key",
            )
            logger.info(f"Generated default API key: {default_key['key']}")
        
        app.config["api_key_manager"] = api_key_manager
        
    except Exception as e:
        logger.warning(f"Security components not available: {e}")
    
    # Try to load validation pipeline
    try:
        from ..filters import ValidationPipeline
        
        validation_pipeline = ValidationPipeline()
        app.config["validation_pipeline"] = validation_pipeline
        logger.info("Image validation pipeline loaded")
        
    except Exception as e:
        logger.warning(f"Validation pipeline not available: {e}")
    
    # Try to load inference engine
    if model_path:
        try:
            from ..engine import InferenceEngine
            
            inference_engine = InferenceEngine(
                model_path=model_path,
                model_format=model_format,
            )
            app.config["inference_engine"] = inference_engine
            logger.info(f"Inference engine loaded: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load inference engine: {e}")
    else:
        logger.warning("No model path provided, inference not available")
    
    # Initialize middleware
    init_middleware(app, api_key_manager, security_headers)
    
    # Register blueprints
    app.register_blueprint(api)
    
    # Root endpoint
    @app.route("/")
    def root():
        return {
            "name": "Sugarcane Pest Detection API",
            "version": app.config["version"],
            "status": "running",
            "documentation": "/api/v1/docs",
        }
    
    logger.info("Flask application created successfully")
    
    return app


def run_app(
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
    model_path: Optional[str] = None,
    model_format: Optional[str] = None,
):
    """
    Run the Flask application.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
        model_path: Path to the model file
        model_format: Model format
    """
    app = create_app(model_path=model_path, model_format=model_format)
    
    logger.info(f"Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Flask Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--format", choices=["pytorch", "onnx", "tflite"], 
                        help="Model format (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    run_app(
        host=args.host,
        port=args.port,
        debug=args.debug,
        model_path=args.model,
        model_format=args.format,
    )
