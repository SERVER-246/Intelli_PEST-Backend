#!/usr/bin/env python3
"""
Simple server startup script with model loading.
"""
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging with timestamps
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)

# Available models
# Primary model - this is where ghost_trainer deploys updated weights
MODELS = {
    "pytorch": r"D:\KnowledgeDistillation\student_model_rotation_robust.pt",  # Ghost trainer target
    "pytorch_12class_backup": r"D:\KnowledgeDistillation\student_model_12class_proper.pt",  # Backup 12-class
    "pytorch_11class": r"D:\KnowledgeDistillation\student_model_final.pth",  # Original 11-class backup
    "onnx": r"D:\KnowledgeDistillation\exported_models\student_model_rotation_robust.onnx",
}

# Broken models (for reference - DO NOT USE)
BROKEN_MODELS = {
    "v1.0.1_broken": r"D:\KnowledgeDistillation\student_model_v1.0.1.pt",  # aux_classifiers mismatch
}

# Default model format
DEFAULT_FORMAT = "pytorch"
MODEL_PATH = MODELS[DEFAULT_FORMAT]

# Class names for feedback system (12 classes including junk)
CLASS_NAMES = [
    "Healthy",
    "Internode borer",
    "Pink borer",
    "Rat damage",
    "Stalk borer",
    "Top borer",
    "army worm",
    "mealy bug",
    "porcupine damage",
    "root borer",
    "termite",
    "junk",  # Filter class for unrelated images
]

if __name__ == "__main__":
    import uvicorn
    from inference_server.fastapi_app.main import create_app
    from inference_server.fastapi_app.dependencies import set_api_key_manager
    from inference_server.security import APIKeyManager
    from inference_server.feedback import init_feedback_manager, init_user_tracker, init_data_collector
    from inference_server.feedback.database import init_database_manager
    from inference_server.training import get_retrain_manager
    
    print("=" * 60)
    print("SUGARCANE PEST DETECTION INFERENCE SERVER")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Format: {DEFAULT_FORMAT}")
    
    # Use a fixed default API key for testing
    DEFAULT_API_KEY = "ip_test_key_intelli_pest_2025"
    
    # Create API manager and register the default key
    api_manager = APIKeyManager()
    api_manager.register_static_key(DEFAULT_API_KEY, tier="admin", name="default_test_key")
    set_api_key_manager(api_manager)
    
    # Initialize feedback and tracking systems
    feedback_dir = os.path.join(os.path.dirname(__file__), "feedback_data")
    
    # Initialize SQLite database (MUST BE FIRST)
    db_path = os.path.join(feedback_dir, "intellipest.db")
    db_manager = init_database_manager(db_path=db_path)
    print(f"Database: {db_path}")
    
    # Log server start
    db_manager.log_system_event("server_start", "main", "Inference server starting",
                                event_data={"model_path": MODEL_PATH, "api_key": DEFAULT_API_KEY[:20] + "..."})
    
    # Initialize user tracker
    user_tracker = init_user_tracker(
        data_dir=os.path.join(feedback_dir, "users")
    )
    print(f"User tracker: {os.path.join(feedback_dir, 'users')}")
    
    # Initialize data collector
    data_collector = init_data_collector(
        base_dir=feedback_dir,
        class_names=CLASS_NAMES,
    )
    print(f"Data collector: {feedback_dir}")
    
    # Initialize feedback manager
    feedback_mgr = init_feedback_manager(
        feedback_dir=feedback_dir,
        images_dir=os.path.join(feedback_dir, "legacy_images"),
        save_images=False,  # Data collector handles this now
        class_names=CLASS_NAMES,
    )
    print(f"Feedback system: {feedback_dir}")
    
    # Initialize retraining manager with auto-scheduler
    print("Initializing retraining system...")
    retrain_manager = get_retrain_manager(
        model_path=MODEL_PATH,
        feedback_dir=os.path.join(feedback_dir, "images"),
    )
    
    # Start the auto-retraining scheduler (checks every 5 minutes)
    retrain_manager.start_auto_scheduler(check_interval_minutes=5)
    print(f"Retraining scheduler: ACTIVE (checks every 5 min)")
    print(f"  Thresholds: {retrain_manager.config.min_images_per_class}/class or {retrain_manager.config.min_total_images} total")
    status = retrain_manager.get_status()
    print(f"  Current feedback images: {status.total_feedback_images}")
    print(f"  Ready to retrain: {status.ready_to_retrain}")
    
    print("")
    print("=" * 60)
    print("DEFAULT API KEY (use this for all requests):")
    print(f"  {DEFAULT_API_KEY}")
    print("=" * 60)
    print("")
    
    # Create app with model
    app = create_app(
        model_path=MODEL_PATH,
        model_format=DEFAULT_FORMAT,
    )
    
    print("Starting server on http://0.0.0.0:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Concurrent connections: 100+ (async support enabled)")
    print("=" * 60)
    
    # Configure uvicorn logging with timestamps (using uvicorn's default access format)
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s - %(levelprefix)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": True,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": "%(asctime)s - %(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": True,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        log_config=log_config,
        limit_concurrency=100,  # Support up to 100 concurrent connections
        backlog=2048,  # Connection queue size
        timeout_keep_alive=60,  # Keep connections alive for 60s
        loop="asyncio",  # Use asyncio for better concurrency
    )
