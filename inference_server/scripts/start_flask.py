#!/usr/bin/env python3
"""
Start Flask Server Script
=========================
Start the Flask inference server with configuration options.
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
        description="Start Flask Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_flask.py --model models/student_model.pth
  python start_flask.py --model models/student_model.onnx --format onnx
  python start_flask.py --model models/model.tflite --format tflite --port 5001
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
        default=int(os.environ.get("PORT", "5000")),
        help="Port to bind to (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("DEBUG", "false").lower() == "true",
        help="Enable debug mode",
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
    
    logger.info(f"Starting Flask server...")
    logger.info(f"Model: {model_path}")
    logger.info(f"Format: {args.format or 'auto-detect'}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    
    try:
        from inference_server.flask_app import create_app
        
        app = create_app(
            model_path=str(model_path),
            model_format=args.format,
        )
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
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
