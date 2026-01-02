"""
ONNX Inference
==============
ONNX Runtime inference implementation.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import time

logger = logging.getLogger(__name__)

# Lazy imports
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

ORT_AVAILABLE = False
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    pass

from pathlib import Path


class ONNXInference:
    """
    ONNX Runtime inference engine.
    
    Handles inference for ONNX models.
    """
    
    def __init__(
        self,
        session: Any = None,
        num_classes: int = 11,
        model_path: Optional[Path] = None,
    ):
        """
        Initialize ONNX inference.
        
        Args:
            session: ONNX Runtime InferenceSession (optional if model_path provided)
            num_classes: Number of output classes
            model_path: Path to ONNX model file (optional if session provided)
        """
        if not ORT_AVAILABLE:
            raise ImportError("ONNX Runtime not installed")
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy not installed")
        
        # Load from path if provided
        if model_path is not None:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            logger.info(f"Loading ONNX model from: {model_path}")
            
            # Get available providers and select best available
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX Runtime providers: {available_providers}")
            
            # Prefer GPU providers, fallback to CPU
            preferred_providers = []
            for provider in ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'DmlExecutionProvider']:
                if provider in available_providers:
                    preferred_providers.append(provider)
            
            # Always add CPU as fallback
            if 'CPUExecutionProvider' in available_providers:
                preferred_providers.append('CPUExecutionProvider')
            
            if not preferred_providers:
                preferred_providers = available_providers  # Use whatever is available
            
            logger.info(f"Using ONNX providers: {preferred_providers}")
            
            session = ort.InferenceSession(
                str(model_path),
                providers=preferred_providers
            )
        
        if session is None:
            raise ValueError("Either session or model_path must be provided")
        
        self.session = session
        self.num_classes = num_classes
        self.providers_used = session.get_providers()
        
        # Get input/output info
        self.input_name = session.get_inputs()[0].name
        self.input_shape = session.get_inputs()[0].shape
        self.output_names = [o.name for o in session.get_outputs()]
        
        logger.info(f"ONNX model loaded with providers: {self.providers_used}")
        logger.info(f"ONNX model: input={self.input_name}, shape={self.input_shape}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB
            
        Returns:
            Preprocessed array (1, C, H, W)
        """
        # Convert to float32
        if image.dtype == np.uint8:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)
        
        # Handle grayscale
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img = (img - mean) / std
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def predict(
        self,
        image: np.ndarray,
        return_features: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on an image.
        
        Args:
            image: Input image as numpy array
            return_features: Whether to return feature vectors
            
        Returns:
            Dictionary with predictions
        """
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Process outputs
        logits = outputs[0]
        
        # Check if output is already probabilities (softmax applied)
        if np.all(logits >= 0) and np.allclose(np.sum(logits, axis=-1), 1.0, atol=0.01):
            probabilities = logits
        else:
            probabilities = self._softmax(logits)
        
        # Flatten if needed
        probabilities = probabilities.flatten()
        logits = logits.flatten()
        
        # Get prediction
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        inference_time = (time.time() - start_time) * 1000
        
        result = {
            "logits": logits,
            "probabilities": probabilities,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "inference_time_ms": inference_time,
        }
        
        # Include features if available and requested
        if return_features and len(outputs) > 1:
            result["features"] = outputs[1].flatten()
        
        return result
    
    def predict_batch(
        self,
        images: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        start_time = time.time()
        
        # Preprocess all images
        batch = np.concatenate([self.preprocess(img) for img in images], axis=0)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: batch})
        logits = outputs[0]
        
        # Process outputs
        if np.all(logits >= 0) and np.allclose(np.sum(logits, axis=-1), 1.0, atol=0.01):
            probabilities = logits
        else:
            probabilities = self._softmax(logits)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Build results
        results = []
        for i in range(len(images)):
            probs = probabilities[i].flatten()
            predicted_class = int(np.argmax(probs))
            
            results.append({
                "logits": logits[i].flatten(),
                "probabilities": probs,
                "predicted_class": predicted_class,
                "confidence": float(probs[predicted_class]),
                "inference_time_ms": inference_time / len(images),
            })
        
        return results
