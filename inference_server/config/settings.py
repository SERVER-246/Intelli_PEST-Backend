"""
Application Settings
====================
Centralized configuration management using Pydantic.
Loads from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from functools import lru_cache
import secrets

# Try to import pydantic, fallback to dataclass if not available
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    try:
        # Try older pydantic version
        from pydantic import BaseSettings, Field, validator as field_validator
        PYDANTIC_AVAILABLE = True
    except ImportError:
        PYDANTIC_AVAILABLE = False
        BaseSettings = object
        Field = lambda **kwargs: kwargs.get("default", kwargs.get("default_factory", lambda: None)())
        field_validator = lambda *args, **kwargs: lambda f: f


class Settings(BaseSettings if PYDANTIC_AVAILABLE else object):
    """Application settings with environment variable support."""
    
    # ===================
    # Application Info
    # ===================
    APP_NAME: str = "Intelli-PEST Inference Server"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = Field(default="production", description="development, staging, production")
    
    # ===================
    # Server Configuration
    # ===================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False
    
    # ===================
    # Paths
    # ===================
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    MODEL_DIR: Optional[Path] = None
    LOG_DIR: Optional[Path] = None
    TEMP_DIR: Optional[Path] = None
    
    def __init__(self, **kwargs):
        if PYDANTIC_AVAILABLE:
            super().__init__(**kwargs)
        else:
            # Manual initialization without pydantic
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        # Set default paths after initialization
        if self.MODEL_DIR is None:
            self.MODEL_DIR = Path(__file__).parent.parent / "models"
        if self.LOG_DIR is None:
            self.LOG_DIR = Path(__file__).parent.parent / "logs"
        if self.TEMP_DIR is None:
            self.TEMP_DIR = Path(__file__).parent.parent / "temp"
    
    # ===================
    # Model Configuration
    # ===================
    DEFAULT_MODEL_FORMAT: str = "onnx"  # onnx, pytorch, tflite
    STUDENT_MODEL_NAME: str = "student_model"
    NUM_CLASSES: int = 11
    IMAGE_SIZE: int = 224
    # Class names - MUST match training config order!
    CLASS_NAMES: List[str] = [
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
        "termite"
    ]
    
    # ===================
    # Inference Settings
    # ===================
    USE_GPU: bool = True
    GPU_DEVICE_ID: int = 0
    BATCH_SIZE_LIMIT: int = 10
    INFERENCE_TIMEOUT: int = 30  # seconds
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # ===================
    # Security Settings
    # ===================
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY_MIN_LENGTH: int = 32
    ADMIN_API_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100  # requests per window
    RATE_LIMIT_WINDOW: int = 60  # seconds
    RATE_LIMIT_BURST: int = 20  # burst allowance
    
    # CORS
    CORS_ENABLED: bool = True
    CORS_ORIGINS: List[str] = ["*"]  # Restrict in production
    CORS_METHODS: List[str] = ["GET", "POST", "OPTIONS"]
    CORS_HEADERS: List[str] = ["*"]
    
    # ===================
    # Image Validation
    # ===================
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    MIN_FILE_SIZE: int = 1024  # 1 KB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    ALLOWED_MIME_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    
    MIN_IMAGE_DIMENSION: int = 64
    MAX_IMAGE_DIMENSION: int = 4096
    MIN_ASPECT_RATIO: float = 0.25
    MAX_ASPECT_RATIO: float = 4.0
    
    # Content Filtering
    CONTENT_FILTER_ENABLED: bool = True
    RELEVANCE_THRESHOLD: float = 0.50
    OOD_DETECTION_ENABLED: bool = True
    OOD_THRESHOLD: float = 0.70
    
    # ===================
    # Logging
    # ===================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_TO_FILE: bool = True
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: int = 30  # days
    
    # Audit Logging
    AUDIT_ENABLED: bool = True
    AUDIT_LOG_REQUESTS: bool = True
    AUDIT_LOG_RESPONSES: bool = True
    AUDIT_SENSITIVE_FIELDS: List[str] = ["api_key", "password", "token"]
    
    # ===================
    # ngrok Configuration
    # ===================
    NGROK_ENABLED: bool = True
    NGROK_AUTH_TOKEN: Optional[str] = None
    NGROK_REGION: str = "us"  # us, eu, ap, au, sa, jp, in
    
    # ===================
    # Docker Configuration
    # ===================
    DOCKER_USER: str = "appuser"
    DOCKER_UID: int = 1000
    DOCKER_GID: int = 1000
    
    if PYDANTIC_AVAILABLE:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = True
            extra = "ignore"


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None


# Convenience function to create directories
def ensure_directories():
    """Create necessary directories if they don't exist."""
    settings = get_settings()
    for dir_path in [settings.MODEL_DIR, settings.LOG_DIR, settings.TEMP_DIR]:
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
