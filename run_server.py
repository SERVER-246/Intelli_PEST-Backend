#!/usr/bin/env python3
"""
Simple server startup script with model loading.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Available models - use the rotation-robust exported models
MODELS = {
    "pytorch": r"D:\KnowledgeDistillation\student_model_rotation_robust.pt",
    "onnx": r"D:\KnowledgeDistillation\exported_models\student_model_rotation_robust.onnx",
}

# Legacy models (kept for reference)
LEGACY_MODELS = {
    "pytorch": r"D:\KnowledgeDistillation\exported_models\student_model.pt",
    "onnx": r"D:\KnowledgeDistillation\exported_models\student_model.onnx",
}

# Default model format
DEFAULT_FORMAT = "pytorch"
MODEL_PATH = MODELS[DEFAULT_FORMAT]

# Class names for feedback system
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
]

if __name__ == "__main__":
    import uvicorn
    from inference_server.fastapi_app.main import create_app
    from inference_server.fastapi_app.dependencies import set_api_key_manager
    from inference_server.security import APIKeyManager
    from inference_server.feedback import init_feedback_manager, init_user_tracker, init_data_collector
    
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
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
