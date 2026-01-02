"""
Model Loader
============
Multi-format model loading supporting PyTorch, ONNX, and TFLite.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """Supported model formats."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TFLITE = "tflite"
    
    @classmethod
    def from_extension(cls, ext: str) -> Optional["ModelFormat"]:
        """Get format from file extension."""
        ext = ext.lower().lstrip(".")
        mapping = {
            "pt": cls.PYTORCH,
            "pth": cls.PYTORCH,
            "onnx": cls.ONNX,
            "tflite": cls.TFLITE,
        }
        return mapping.get(ext)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    format: ModelFormat
    path: Path
    num_classes: int
    input_size: Tuple[int, int]
    device: str
    loaded: bool = False
    load_time_ms: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelLoader:
    """
    Universal model loader supporting multiple formats.
    
    Automatically detects format from file extension and loads
    using the appropriate backend.
    """
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        default_device: str = "auto",
        num_classes: int = 11,
        input_size: Tuple[int, int] = (224, 224),
    ):
        """
        Initialize model loader.
        
        Args:
            model_dir: Base directory for models
            default_device: Default device (auto, cpu, cuda)
            num_classes: Number of output classes
            input_size: Expected input size (H, W)
        """
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent.parent / "models"
        self.default_device = default_device
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Loaded models cache
        self._models: Dict[str, Any] = {}
        self._model_info: Dict[str, ModelInfo] = {}
        
        # Detect available device
        self._device = self._detect_device()
    
    def _detect_device(self) -> str:
        """Detect available compute device."""
        if self.default_device != "auto":
            return self.default_device
        
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                return "cuda"
        except ImportError:
            pass
        
        logger.info("Using CPU for inference")
        return "cpu"
    
    def load(
        self,
        model_path: Union[str, Path],
        model_name: Optional[str] = None,
        force_reload: bool = False,
    ) -> Tuple[Any, ModelInfo]:
        """
        Load a model from file.
        
        Args:
            model_path: Path to model file (absolute or relative to model_dir)
            model_name: Optional name for the model
            force_reload: Force reload even if cached
            
        Returns:
            Tuple of (model, ModelInfo)
        """
        # Resolve path
        path = Path(model_path)
        if not path.is_absolute():
            path = self.model_dir / path
        
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        # Get model name
        name = model_name or path.stem
        
        # Check cache
        if name in self._models and not force_reload:
            logger.debug(f"Using cached model: {name}")
            return self._models[name], self._model_info[name]
        
        # Detect format
        fmt = ModelFormat.from_extension(path.suffix)
        if fmt is None:
            raise ValueError(f"Unsupported model format: {path.suffix}")
        
        logger.info(f"Loading {fmt.value} model: {path}")
        start_time = time.time()
        
        # Load based on format
        if fmt == ModelFormat.PYTORCH:
            model = self._load_pytorch(path)
        elif fmt == ModelFormat.ONNX:
            model = self._load_onnx(path)
        elif fmt == ModelFormat.TFLITE:
            model = self._load_tflite(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        
        load_time = (time.time() - start_time) * 1000
        
        # Create model info
        info = ModelInfo(
            name=name,
            format=fmt,
            path=path,
            num_classes=self.num_classes,
            input_size=self.input_size,
            device=self._device,
            loaded=True,
            load_time_ms=load_time,
        )
        
        # Cache
        self._models[name] = model
        self._model_info[name] = info
        
        logger.info(f"Model '{name}' loaded in {load_time:.2f}ms")
        return model, info
    
    def _load_pytorch(self, path: Path) -> Any:
        """Load PyTorch model."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        # Try loading as state dict first, then as full model
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location=self._device)
            
            # Check if it's a state dict or full model
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    # Need model architecture to load state dict
                    logger.warning("State dict found but no architecture. Loading raw checkpoint.")
                    return checkpoint
                elif "state_dict" in checkpoint:
                    return checkpoint
            
            # It's a full model
            model = checkpoint
            if hasattr(model, 'eval'):
                model.eval()
            if hasattr(model, 'to'):
                model = model.to(self._device)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def _load_onnx(self, path: Path) -> Any:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime-gpu")
        
        # Set up providers based on device
        if self._device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        try:
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            session = ort.InferenceSession(
                str(path),
                sess_options=sess_options,
                providers=providers,
            )
            
            logger.info(f"ONNX session created with providers: {session.get_providers()}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _load_tflite(self, path: Path) -> Any:
        """Load TFLite model."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        try:
            interpreter = tf.lite.Interpreter(model_path=str(path))
            interpreter.allocate_tensors()
            
            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            logger.info(f"TFLite model loaded. Input: {input_details[0]['shape']}, Output: {output_details[0]['shape']}")
            
            return interpreter
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            raise
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        return self._models.get(name)
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self._model_info.get(name)
    
    def list_models(self) -> Dict[str, ModelInfo]:
        """List all loaded models."""
        return self._model_info.copy()
    
    def unload(self, name: str) -> bool:
        """Unload a model from cache."""
        if name in self._models:
            del self._models[name]
            del self._model_info[name]
            logger.info(f"Model '{name}' unloaded")
            return True
        return False
    
    def unload_all(self):
        """Unload all models."""
        self._models.clear()
        self._model_info.clear()
        logger.info("All models unloaded")
    
    @property
    def device(self) -> str:
        """Get current device."""
        return self._device


# Global loader instance
_loader: Optional[ModelLoader] = None


def get_model_loader(**kwargs) -> ModelLoader:
    """Get or create the global model loader."""
    global _loader
    if _loader is None:
        _loader = ModelLoader(**kwargs)
    return _loader


def load_model(
    model_path: Union[str, Path],
    model_name: Optional[str] = None,
) -> Tuple[Any, ModelInfo]:
    """Load a model using the global loader."""
    return get_model_loader().load(model_path, model_name)
