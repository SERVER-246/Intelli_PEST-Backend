#!/usr/bin/env python3
"""
Start FastAPI Server Script
===========================
Start the FastAPI inference server with configuration options.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Start FastAPI Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_fastapi.py --model models/student_model.pth
  python start_fastapi.py --model models/student_model.onnx --format onnx
  python start_fastapi.py --model models/model.tflite --port 8080 --workers 4
        """,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_PATH"),
        help="Path to model file (required)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pytorch", "onnx", "tflite"],
        default=os.environ.get("MODEL_FORMAT"),
        help="Model format (auto-detected if not specified)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "1")),
        help="Number of workers (default: 1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.environ.get("RELOAD", "false").lower() == "true",
        help="Enable auto-reload (development only)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate model path
    if not args.model:
        logger.error("Model path is required. Use --model or set MODEL_PATH environment variable.")
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    logger.info(f"Starting FastAPI server...")
    logger.info(f"Model: {model_path}")
    logger.info(f"Format: {args.format or 'auto-detect'}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Workers: {args.workers}")
    
    try:
        import uvicorn
        from inference_server.fastapi_app import create_app
        
        # Create app with model configuration
        # Note: For production with workers > 1, use environment variables
        os.environ["MODEL_PATH"] = str(model_path)
        if args.format:
            os.environ["MODEL_FORMAT"] = args.format
        
        app = create_app(
            model_path=str(model_path),
            model_format=args.format,
        )
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level=args.log_level.lower(),
        )
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
