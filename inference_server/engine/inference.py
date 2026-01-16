"""
Unified Inference Engine
========================
High-level inference interface that abstracts model format differences.

Includes Phase 3 integration for advanced capabilities:
- Region-aware perception
- Multi-label classification
- Attention maps for explainability
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
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

# Phase 3 integration (optional)
try:
    import sys
    # Add black_ops_training to path for Phase 3 imports
    _BLACK_OPS_DIR = Path(__file__).parent.parent.parent / "black_ops_training"
    if _BLACK_OPS_DIR.exists() and str(_BLACK_OPS_DIR) not in sys.path:
        sys.path.insert(0, str(_BLACK_OPS_DIR))
    
    from phase3_enabled_config import (  # pyright: ignore
        Phase3ProductionMode,
        enable_phase3,
        disable_phase3_completely,
        get_phase3_status,
        create_inference_phase3_config,
    )
    from phase3_integration import (  # pyright: ignore
        Phase3Configuration,
        Phase3IntegrationManager,
        Phase3SafeIntegrationWrapper,
        Phase3IntegrationResult,
        initialize_phase3,
        get_phase3_manager,
    )
    from phase3_dormant_components import Phase3DormancyFlags  # pyright: ignore
    PHASE3_AVAILABLE = True
    logger.info("Phase 3 integration available")
except ImportError as e:
    PHASE3_AVAILABLE = False
    logger.debug(f"Phase 3 not available: {e}")


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
    
    # Phase 3 extended results
    phase3_enabled: bool = False
    phase3_result: Optional[Dict[str, Any]] = None
    multi_label_predictions: Optional[List[Dict[str, Any]]] = None
    region_scores: Optional[Dict[int, float]] = None
    attention_map: Optional[Any] = None
    
    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for backward compatibility."""
        # Map common keys to dataclass fields
        key_map = {
            "class_name": self.class_name,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "logits": self.logits,
            "features": self.features,
            "inference_time_ms": self.inference_time_ms,
            "model_name": self.model_name,
            "model_format": self.model_format,
            "device": self.device,
            "error": self.error,
            "success": self.success,
            "phase3": self.phase3_result,
            "phase3_enabled": self.phase3_enabled,
        }
        return key_map.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Dict-like access for backward compatibility."""
        result = self.get(key)
        if result is None and key not in ["error", "phase3", "features", "logits"]:
            raise KeyError(key)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
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
        
        # Add Phase 3 results if available
        if self.phase3_enabled and self.phase3_result:
            result["phase3"] = {
                "enabled": True,
                "multi_label": self.multi_label_predictions,
                "regions": {
                    "scores": self.region_scores,
                    "num_regions": len(self.region_scores) if self.region_scores else 0,
                } if self.region_scores else None,
                "has_attention_map": self.attention_map is not None,
            }
        
        return result


class InferenceEngine:
    """
    Unified inference engine supporting multiple model formats.
    
    Provides a consistent interface regardless of underlying model format.
    Includes Phase 3 integration for advanced capabilities.
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
        enable_phase3: bool = True,
        phase3_mode: str = "inference",
    ):
        """
        Initialize inference engine.
        
        Args:
            model_dir: Base directory for models
            default_format: Default model format to use
            device: Device for inference (auto, cpu, cuda)
            class_names: List of class names
            enable_phase3: Whether to enable Phase 3 capabilities
            phase3_mode: Phase 3 mode ("inference", "evaluation", "full")
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for inference")
        
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent.parent / "models"
        self.default_format = default_format
        self.device = device
        self.class_names = class_names or self.CLASS_NAMES
        
        # Phase 3 integration
        self._phase3_enabled = False
        self._phase3_manager = None
        self._phase3_wrapper = None
        
        if enable_phase3 and PHASE3_AVAILABLE:
            self._initialize_phase3(phase3_mode)
        
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
    
    @property
    def model_format(self) -> str:
        """Return current model format (for API compatibility)."""
        return self._current_format or self.default_format
    
    def _initialize_phase3(self, mode: str = "inference"):
        """Initialize Phase 3 capabilities."""
        if not PHASE3_AVAILABLE:
            return
        
        try:
            mode_map = {
                "inference": Phase3ProductionMode.INFERENCE,
                "evaluation": Phase3ProductionMode.EVALUATION,
                "full": Phase3ProductionMode.FULL,
                "minimal": Phase3ProductionMode.MINIMAL,
            }
            p3_mode = mode_map.get(mode, Phase3ProductionMode.INFERENCE)
            
            self._phase3_manager = enable_phase3(p3_mode)
            if self._phase3_manager:
                self._phase3_enabled = True
                logger.info(f"Phase 3 enabled: {mode} mode")
        except Exception as e:
            logger.warning(f"Phase 3 initialization failed: {e}")
            self._phase3_enabled = False
    
    def _run_phase3_inference(
        self,
        model: Any,
        image_tensor: Any,
        logits: Any,
        predicted_class: int
    ) -> Optional[Dict[str, Any]]:
        """Run Phase 3 inference if enabled."""
        if not self._phase3_enabled or not self._phase3_manager:
            return None
        
        try:
            result = self._phase3_manager.run_inference(
                model, image_tensor, logits, predicted_class
            )
            
            if result.is_empty():
                return None
            
            # Convert to dict format expected by routers.py
            phase3_dict = {
                "executed": result.phase3_executed,
                "processing_time_ms": result.execution_time_ms,
                "had_failure": result.had_failure,
                "error": result.failure_message if result.had_failure else None,
            }
            
            # Regions - convert to list format
            if not result.regions.is_empty():
                regions_list = []
                # Build regions with relevance scores if available
                if not result.relevance_scores.is_empty():
                    for region_id, score in result.relevance_scores.region_scores.items():
                        regions_list.append({
                            "region_id": region_id,
                            "relevance_score": float(score),
                            "bbox": None,  # Phase 3 grid doesn't provide bbox
                            "label": None,
                        })
                    phase3_dict["regions"] = regions_list
                    phase3_dict["top_region_score"] = max(
                        result.relevance_scores.region_scores.values()
                    ) if result.relevance_scores.region_scores else None
            
            # Multi-label predictions - convert to expected format
            if not result.multi_label.is_empty():
                predictions = []
                for i, label in enumerate(result.multi_label.predicted_labels):
                    conf = result.multi_label.label_confidences[i] if i < len(result.multi_label.label_confidences) else 0.0
                    predictions.append({
                        "label": label,
                        "confidence": float(conf),
                    })
                phase3_dict["multi_label"] = {"predictions": predictions}
            
            # Attention map
            if not result.attention_map.is_empty():
                phase3_dict["attention_map"] = result.attention_map.attention_map_base64 if hasattr(result.attention_map, 'attention_map_base64') else None
                phase3_dict["attention_method"] = result.attention_map.generation_method
            
            return phase3_dict
        except Exception as e:
            logger.debug(f"Phase 3 inference error: {e}")
            return None
    
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
        image: Any,  # Can be np.ndarray or bytes
        return_features: bool = False,
    ) -> InferenceResult:
        """
        Run inference on a single image.
        
        Args:
            image: Input image as numpy array (H, W, C), bytes, or PIL Image
            return_features: Whether to return feature vectors
            
        Returns:
            InferenceResult with prediction
        """
        try:
            engine, info = self._get_engine()
            
            # Run inference (underlying engine handles bytes/ndarray conversion)
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
            
            # Run Phase 3 if enabled (non-blocking)
            phase3_result = None
            multi_label_preds = None
            region_scores = None
            attention_map = None
            
            if self._phase3_enabled:
                try:
                    # Get model from engine for Phase 3
                    model = getattr(engine, 'model', None)
                    image_tensor = result.get("image_tensor")  # PyTorch tensor for Phase 3
                    # Prefer tensor version of logits if available
                    logits = result.get("logits_tensor", result.get("logits"))
                    
                    if model is not None and image_tensor is not None:
                        phase3_result = self._run_phase3_inference(
                            model, image_tensor, logits, predicted_class
                        )
                        
                        if phase3_result:
                            if phase3_result.get("multi_label"):
                                multi_label_preds = phase3_result["multi_label"].get("predictions")
                            if phase3_result.get("relevance_scores"):
                                region_scores = phase3_result["relevance_scores"]
                    else:
                        logger.debug(f"Phase 3 skipped: model={model is not None}, tensor={image_tensor is not None}")
                except Exception as p3_err:
                    logger.debug(f"Phase 3 post-processing skipped: {p3_err}")
            
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
                phase3_enabled=self._phase3_enabled,
                phase3_result=phase3_result,
                multi_label_predictions=multi_label_preds,
                region_scores=region_scores,
                attention_map=attention_map,
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
