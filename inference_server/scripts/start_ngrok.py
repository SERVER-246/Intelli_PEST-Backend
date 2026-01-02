#!/usr/bin/env python3
"""
Start with Ngrok Script
=======================
Start the inference server with ngrok tunnel for public access.
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def start_ngrok(port: int, auth_token: str = None) -> str:
    """
    Start ngrok tunnel.
    
    Args:
        port: Local port to tunnel
        auth_token: Ngrok auth token
        
    Returns:
        Public URL
    """
    try:
        from pyngrok import ngrok, conf
        
        if auth_token:
            conf.get_default().auth_token = auth_token
        
        # Start tunnel
        public_url = ngrok.connect(port, "http")
        
        return str(public_url)
        
    except ImportError:
        raise ImportError("pyngrok is not installed. Install with: pip install pyngrok")


def main():
    parser = argparse.ArgumentParser(
        description="Start Inference Server with Ngrok Tunnel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_ngrok.py --model models/student_model.pth
  python start_ngrok.py --model models/model.onnx --framework fastapi
  python start_ngrok.py --model models/model.pth --auth-token YOUR_NGROK_TOKEN
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
        "--framework",
        type=str,
        choices=["flask", "fastapi"],
        default="fastapi",
        help="Web framework to use (default: fastapi)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Local port (default: 8000)",
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default=os.environ.get("NGROK_AUTHTOKEN"),
        help="Ngrok authentication token",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
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
    
    # Start ngrok
    logger.info("Starting ngrok tunnel...")
    try:
        public_url = start_ngrok(args.port, args.auth_token)
        logger.info(f"=" * 60)
        logger.info(f"NGROK PUBLIC URL: {public_url}")
        logger.info(f"=" * 60)
        logger.info(f"Share this URL with your Android app!")
        logger.info(f"API Endpoint: {public_url}/api/v1/predict")
        logger.info(f"Health Check: {public_url}/api/v1/health")
        logger.info(f"Documentation: {public_url}/docs (FastAPI only)")
        logger.info(f"=" * 60)
    except ImportError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start ngrok: {e}")
        sys.exit(1)
    
    # Start server
    logger.info(f"Starting {args.framework} server on port {args.port}...")
    
    try:
        if args.framework == "fastapi":
            import uvicorn
            from inference_server.fastapi_app import create_app
            
            app = create_app(
                model_path=str(model_path),
                model_format=args.format,
            )
            
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=args.port,
                log_level=args.log_level.lower(),
            )
        else:
            from inference_server.flask_app import create_app
            
            app = create_app(
                model_path=str(model_path),
                model_format=args.format,
            )
            
            app.run(
                host="0.0.0.0",
                port=args.port,
                debug=False,
                threaded=True,
            )
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)
    finally:
        # Cleanup ngrok
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except:
            pass


if __name__ == "__main__":
    main()
