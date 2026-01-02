"""
Model Registry
==============
Manages available models and their configurations.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import yaml
import json

from .model_loader import ModelFormat

logger = logging.getLogger(__name__)


@dataclass
class RegisteredModel:
    """Information about a registered model."""
    name: str
    display_name: str
    description: str = ""
    version: str = "1.0.0"
    exposed: bool = False  # Whether model is available via public API
    formats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accuracy: Optional[float] = None
    parameters: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_path(self, format: str, base_dir: Path) -> Optional[Path]:
        """Get model file path for a specific format."""
        if format not in self.formats:
            return None
        
        filename = self.formats[format].get("filename")
        if not filename:
            return None
        
        # Check in format-specific subdirectory or model directory
        paths_to_check = [
            base_dir / self.name / filename,
            base_dir / filename,
            base_dir / format / filename,
        ]
        
        for path in paths_to_check:
            if path.exists():
                return path
        
        return base_dir / self.name / filename  # Return expected path
    
    def available_formats(self) -> List[str]:
        """Get list of available formats."""
        return list(self.formats.keys())
    
    def to_dict(self, include_paths: bool = False, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "version": self.version,
            "exposed": self.exposed,
            "formats": list(self.formats.keys()),
            "accuracy": self.accuracy,
            "parameters": self.parameters,
        }
        
        if include_paths and base_dir:
            data["paths"] = {
                fmt: str(self.get_path(fmt, base_dir))
                for fmt in self.formats
            }
        
        return data


class ModelRegistry:
    """
    Registry for managing available models.
    
    Tracks which models are available, their formats, and metadata.
    """
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize model registry.
        
        Args:
            model_dir: Base directory for models
            config_path: Path to model configuration YAML
        """
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent.parent / "models"
        self.config_path = config_path
        
        self._models: Dict[str, RegisteredModel] = {}
        
        # Load configuration if provided
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
        else:
            # Load default configuration
            default_config = Path(__file__).parent.parent / "config" / "model_config.yaml"
            if default_config.exists():
                self._load_config(default_config)
    
    def _load_config(self, config_path: Path):
        """Load model configuration from YAML."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Register student model
            if "student_model" in config:
                student = config["student_model"]
                self.register(RegisteredModel(
                    name="student",
                    display_name=student.get("name", "Student Model"),
                    description=student.get("description", ""),
                    version=student.get("version", "1.0.0"),
                    exposed=student.get("exposed", True),
                    formats=student.get("formats", {}),
                    accuracy=student.get("accuracy"),
                    parameters=student.get("parameters"),
                ))
            
            # Register teacher models
            if "teacher_models" in config:
                for model_id, model_config in config["teacher_models"].items():
                    self.register(RegisteredModel(
                        name=model_id,
                        display_name=model_config.get("name", model_id),
                        description=model_config.get("description", ""),
                        version=model_config.get("version", "1.0.0"),
                        exposed=model_config.get("exposed", False),
                        formats={fmt: {} for fmt in model_config.get("formats", [])},
                    ))
            
            logger.info(f"Loaded {len(self._models)} models from config")
            
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
    
    def register(self, model: RegisteredModel):
        """Register a model."""
        self._models[model.name] = model
        logger.debug(f"Registered model: {model.name}")
    
    def unregister(self, name: str) -> bool:
        """Unregister a model."""
        if name in self._models:
            del self._models[name]
            logger.debug(f"Unregistered model: {name}")
            return True
        return False
    
    def get(self, name: str) -> Optional[RegisteredModel]:
        """Get a registered model by name."""
        return self._models.get(name)
    
    def get_student_model(self) -> Optional[RegisteredModel]:
        """Get the student model."""
        return self._models.get("student")
    
    def list_all(self) -> List[RegisteredModel]:
        """List all registered models."""
        return list(self._models.values())
    
    def list_exposed(self) -> List[RegisteredModel]:
        """List only publicly exposed models."""
        return [m for m in self._models.values() if m.exposed]
    
    def list_internal(self) -> List[RegisteredModel]:
        """List internal (non-exposed) models."""
        return [m for m in self._models.values() if not m.exposed]
    
    def find_model_path(
        self,
        name: str,
        format: str = "onnx",
    ) -> Optional[Path]:
        """
        Find model file path.
        
        Args:
            name: Model name
            format: Desired format (pytorch, onnx, tflite)
            
        Returns:
            Path to model file or None
        """
        model = self._models.get(name)
        if not model:
            return None
        
        return model.get_path(format, self.model_dir)
    
    def scan_directory(self):
        """
        Scan model directory and update registry with found models.
        """
        if not self.model_dir.exists():
            logger.warning(f"Model directory does not exist: {self.model_dir}")
            return
        
        # Look for model files
        extensions = {".pt", ".pth", ".onnx", ".tflite"}
        
        for path in self.model_dir.rglob("*"):
            if path.suffix.lower() in extensions:
                model_name = path.stem
                fmt = ModelFormat.from_extension(path.suffix)
                
                if fmt and model_name not in self._models:
                    # Auto-register found model
                    self.register(RegisteredModel(
                        name=model_name,
                        display_name=model_name.replace("_", " ").title(),
                        formats={fmt.value: {"filename": path.name}},
                        exposed=False,  # Not exposed by default
                    ))
                    logger.info(f"Auto-registered model: {model_name} ({fmt.value})")
    
    def to_dict(self, exposed_only: bool = True) -> Dict[str, Any]:
        """
        Export registry as dictionary.
        
        Args:
            exposed_only: Only include exposed models
            
        Returns:
            Dictionary representation
        """
        models = self.list_exposed() if exposed_only else self.list_all()
        return {
            "models": [m.to_dict(include_paths=False) for m in models],
            "count": len(models),
        }


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry(**kwargs) -> ModelRegistry:
    """Get or create the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(**kwargs)
    return _registry
