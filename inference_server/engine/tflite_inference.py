"""
TFLite Inference
================
TensorFlow Lite inference implementation.
"""

import logging
from typing import Optional, Dict, Any, List
import time

logger = logging.getLogger(__name__)

# Lazy imports
NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    pass


class TFLiteInference:
    """
    TensorFlow Lite inference engine.
    
    Handles inference for TFLite models.
    """
    
    def __init__(
        self,
        interpreter: Any,
        num_classes: int = 11,
    ):
        """
        Initialize TFLite inference.
        
        Args:
            interpreter: TFLite Interpreter
            num_classes: Number of output classes
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed")
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy not installed")
        
        self.interpreter = interpreter
        self.num_classes = num_classes
        
        # Get input/output details
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        
        # Get input shape and type
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']
        
        # Check for quantization
        self.is_quantized = self.input_dtype == np.uint8 or self.input_dtype == np.int8
        
        if self.is_quantized:
            self.input_scale = self.input_details[0].get('quantization_parameters', {}).get('scales', [1.0])[0]
            self.input_zero_point = self.input_details[0].get('quantization_parameters', {}).get('zero_points', [0])[0]
            self.output_scale = self.output_details[0].get('quantization_parameters', {}).get('scales', [1.0])[0]
            self.output_zero_point = self.output_details[0].get('quantization_parameters', {}).get('zero_points', [0])[0]
        
        logger.info(f"TFLite model: input_shape={self.input_shape}, dtype={self.input_dtype}, quantized={self.is_quantized}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB
            
        Returns:
            Preprocessed array matching model input requirements
        """
        # Get expected shape (batch, height, width, channels) for TFLite
        expected_shape = self.input_shape
        
        # Convert to float32 first
        if image.dtype == np.uint8:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)
        
        # Handle grayscale
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        
        # Check if model expects NHWC or NCHW
        if len(expected_shape) == 4:
            if expected_shape[1] == 3:  # NCHW format
                # Normalize with ImageNet stats first
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
                img = (img - mean) / std
                # Convert HWC to CHW
                img = np.transpose(img, (2, 0, 1))
            else:  # NHWC format
                # Normalize with ImageNet stats
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
                img = (img - mean) / std
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Handle quantized models
        if self.is_quantized:
            img = img / self.input_scale + self.input_zero_point
            img = np.clip(img, 0, 255).astype(self.input_dtype)
        else:
            img = img.astype(self.input_dtype)
        
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
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_index, input_tensor)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_index)
        
        # Dequantize if needed
        if self.is_quantized:
            output = (output.astype(np.float32) - self.output_zero_point) * self.output_scale
        
        logits = output.flatten()
        
        # Check if output is already probabilities
        if np.all(logits >= 0) and np.allclose(np.sum(logits), 1.0, atol=0.01):
            probabilities = logits
        else:
            probabilities = self._softmax(logits)
        
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
        
        # Include features if available
        if return_features and len(self.output_details) > 1:
            features = self.interpreter.get_tensor(self.output_details[1]['index'])
            result["features"] = features.flatten()
        
        return result
    
    def predict_batch(
        self,
        images: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.
        
        Note: TFLite typically processes one image at a time.
        This method processes images sequentially.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        total_start = time.time()
        
        for image in images:
            result = self.predict(image, return_features=False)
            results.append(result)
        
        # Adjust timing to be per-image average
        total_time = (time.time() - total_start) * 1000
        avg_time = total_time / len(images)
        
        for result in results:
            result["inference_time_ms"] = avg_time
        
        return results
