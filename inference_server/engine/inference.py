"""
Unified Inference Engine
========================
High-level inference interface that abstracts model format differences.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass
import time

from .model_loader import ModelLoader, ModelFormat, ModelInfo, get_model_loader
from .model_registry import ModelRegistry, RegisteredModel, get_model_registry
from .pytorch_inference import PyTorchInference
from .onnx_inference import ONNXInference
from .tflite_inference import TFLiteInference

logger = logging.getLogger(__name__)

# Lazy import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class InferenceResult:
    """Result from inference."""
    success: bool
    predicted_class: int = -1
    class_name: str = ""
    confidence: float = 0.0
    probabilities: Dict[str, float] = None
    logits: Any = None
    features: Any = None
    inference_time_ms: float = 0.0
    model_name: str = ""
    model_format: str = ""
    device: str = ""
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "success": self.success,
            "prediction": {
                "class": self.class_name,
                "class_id": self.predicted_class,
                "confidence": self.confidence,
            },
            "probabilities": self.probabilities,
            "inference": {
                "time_ms": self.inference_time_ms,
                "model": self.model_name,
                "format": self.model_format,
                "device": self.device,
            },
            "error": self.error,
        }


class InferenceEngine:
    """
    Unified inference engine supporting multiple model formats.
    
    Provides a consistent interface regardless of underlying model format.
    """
    
    # Class name mapping - ORDER MUST MATCH ImageFolder alphabetical sorting!
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
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        default_format: str = "onnx",
        device: str = "auto",
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_dir: Base directory for models
            default_format: Default model format to use
            device: Device for inference (auto, cpu, cuda)
            class_names: List of class names
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for inference")
        
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent.parent / "models"
        self.default_format = default_format
        self.device = device
        self.class_names = class_names or self.CLASS_NAMES
        
        # Initialize loader and registry
        self.loader = get_model_loader(
            model_dir=self.model_dir,
            default_device=device,
        )
        self.registry = get_model_registry(model_dir=self.model_dir)
        
        # Cache for inference engines
        self._engines: Dict[str, Any] = {}
        
        # Track loaded model
        self._current_model: Optional[str] = None
        self._current_format: Optional[str] = None
    
    def load_model(
        self,
        model_name: str = "student",
        format: Optional[str] = None,
        force_reload: bool = False,
    ) -> bool:
        """
        Load a model for inference.
        
        Args:
            model_name: Name of the model to load
            format: Model format (uses default if None)
            force_reload: Force reload even if already loaded
            
        Returns:
            True if loaded successfully
        """
        format = format or self.default_format
        cache_key = f"{model_name}_{format}"
        
        # Check if already loaded
        if cache_key in self._engines and not force_reload:
            self._current_model = model_name
            self._current_format = format
            logger.debug(f"Using cached model: {cache_key}")
            return True
        
        # Find model path
        model_info = self.registry.get(model_name)
        
        if model_info:
            model_path = model_info.get_path(format, self.model_dir)
        else:
            # Try direct path lookup
            extensions = {"pytorch": [".pt", ".pth"], "onnx": [".onnx"], "tflite": [".tflite"]}
            model_path = None
            for ext in extensions.get(format, []):
                candidate = self.model_dir / f"{model_name}{ext}"
                if candidate.exists():
                    model_path = candidate
                    break
        
        if not model_path or not model_path.exists():
            logger.error(f"Model not found: {model_name} ({format})")
            return False
        
        try:
            # Load model
            model, info = self.loader.load(model_path, model_name, force_reload)
            
            # Create appropriate inference engine
            if info.format == ModelFormat.PYTORCH:
                engine = PyTorchInference(model, device=self.loader.device)
            elif info.format == ModelFormat.ONNX:
                engine = ONNXInference(model)
            elif info.format == ModelFormat.TFLITE:
                engine = TFLiteInference(model)
            else:
                raise ValueError(f"Unsupported format: {info.format}")
            
            # Cache engine
            self._engines[cache_key] = {
                "engine": engine,
                "info": info,
            }
            
            self._current_model = model_name
            self._current_format = format
            
            logger.info(f"Model loaded: {model_name} ({format}) on {self.loader.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def _get_engine(self) -> Tuple[Any, ModelInfo]:
        """Get current inference engine."""
        if not self._current_model or not self._current_format:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        cache_key = f"{self._current_model}_{self._current_format}"
        cached = self._engines.get(cache_key)
        
        if not cached:
            raise RuntimeError(f"Model not found in cache: {cache_key}")
        
        return cached["engine"], cached["info"]
    
    def predict(
        self,
        image: np.ndarray,
        return_features: bool = False,
    ) -> InferenceResult:
        """
        Run inference on a single image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB, 0-255 or 0-1
            return_features: Whether to return feature vectors
            
        Returns:
            InferenceResult with prediction
        """
        try:
            engine, info = self._get_engine()
            
            # Run inference
            result = engine.predict(image, return_features=return_features)
            
            # Build response
            predicted_class = result["predicted_class"]
            class_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else f"class_{predicted_class}"
            
            # Build probabilities dict
            probs = result["probabilities"]
            probabilities = {
                self.class_names[i] if i < len(self.class_names) else f"class_{i}": float(probs[i])
                for i in range(len(probs))
            }
            
            return InferenceResult(
                success=True,
                predicted_class=predicted_class,
                class_name=class_name,
                confidence=result["confidence"],
                probabilities=probabilities,
                logits=result.get("logits"),
                features=result.get("features"),
                inference_time_ms=result["inference_time_ms"],
                model_name=info.name,
                model_format=info.format.value,
                device=info.device,
            )
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return InferenceResult(
                success=False,
                error=str(e),
            )
    
    def predict_batch(
        self,
        images: List[np.ndarray],
    ) -> List[InferenceResult]:
        """
        Run inference on multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of InferenceResult objects
        """
        try:
            engine, info = self._get_engine()
            
            # Run batch inference
            results = engine.predict_batch(images)
            
            # Build responses
            inference_results = []
            for result in results:
                predicted_class = result["predicted_class"]
                class_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else f"class_{predicted_class}"
                
                probs = result["probabilities"]
                probabilities = {
                    self.class_names[i] if i < len(self.class_names) else f"class_{i}": float(probs[i])
                    for i in range(len(probs))
                }
                
                inference_results.append(InferenceResult(
                    success=True,
                    predicted_class=predicted_class,
                    class_name=class_name,
                    confidence=result["confidence"],
                    probabilities=probabilities,
                    logits=result.get("logits"),
                    inference_time_ms=result["inference_time_ms"],
                    model_name=info.name,
                    model_format=info.format.value,
                    device=info.device,
                ))
            
            return inference_results
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return [InferenceResult(success=False, error=str(e)) for _ in images]
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about currently loaded model."""
        if not self._current_model:
            return None
        
        cache_key = f"{self._current_model}_{self._current_format}"
        cached = self._engines.get(cache_key)
        
        if not cached:
            return None
        
        info = cached["info"]
        return {
            "name": info.name,
            "format": info.format.value,
            "path": str(info.path),
            "device": info.device,
            "num_classes": info.num_classes,
            "input_size": info.input_size,
            "load_time_ms": info.load_time_ms,
        }
    
    def list_available_models(self, exposed_only: bool = True) -> List[Dict[str, Any]]:
        """List available models."""
        models = self.registry.list_exposed() if exposed_only else self.registry.list_all()
        return [m.to_dict() for m in models]
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        return self.class_names.copy()
    
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._current_model is not None


# Global engine instance
_engine: Optional[InferenceEngine] = None


def get_inference_engine(**kwargs) -> InferenceEngine:
    """Get or create the global inference engine."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine(**kwargs)
    return _engine


def reset_inference_engine():
    """Reset the global inference engine."""
    global _engine
    _engine = None
