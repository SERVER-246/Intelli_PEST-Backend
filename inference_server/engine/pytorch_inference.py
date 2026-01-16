"""
PyTorch Inference
=================
PyTorch-specific inference implementation.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import time

logger = logging.getLogger(__name__)

# Lazy imports
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    pass

NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass


class PyTorchInference:
    """
    PyTorch inference engine.
    
    Handles inference for PyTorch models (.pt, .pth files).
    """
    
    # Default class names - MUST match training config order!
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
        model: Any = None,
        model_path: Path = None,
        device: str = "auto",
        num_classes: int = 11,
        half_precision: bool = False,
        class_names: List[str] = None,
    ):
        """
        Initialize PyTorch inference.
        
        Args:
            model: Loaded PyTorch model (optional)
            model_path: Path to model file (optional)
            device: Device to run inference on
            num_classes: Number of output classes
            half_precision: Use FP16 inference
            class_names: List of class names
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.num_classes = num_classes
        self.half_precision = half_precision
        self.class_names = class_names or self.CLASS_NAMES
        self.model_format = "pytorch"
        self._is_state_dict = False
        
        # Load model from path if provided
        if model_path is not None:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            logger.info(f"Loading PyTorch model from {model_path}")
            model = self._load_model_file(model_path)
        
        # Handle different model types
        if model is None:
            raise ValueError("Either model or model_path must be provided")
        
        if isinstance(model, dict):
            # It's a state dict or checkpoint
            self.model = model
            self._is_state_dict = True
            logger.warning("Model is state dict, direct inference not supported")
        else:
            self.model = model
            self._is_state_dict = False
            
            # Move to device and set to eval mode
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Apply half precision if requested
            if half_precision and self.device == "cuda":
                self.model = self.model.half()
        
        logger.info(f"PyTorchInference initialized on {self.device}")
    
    def _load_model_file(self, path: Path) -> Any:
        """Load model from file."""
        try:
            # First try loading as a full model
            model = torch.load(path, map_location=self.device)
            
            # Check if it's a checkpoint with model key
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'model_state_dict' in model:
                    # State dict only - need architecture
                    logger.warning("Checkpoint contains only state_dict, not full model")
                    # Try to create model architecture and load weights
                    return self._create_model_from_state_dict(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _create_model_from_state_dict(self, checkpoint: dict) -> Any:
        """Create model from state dict - uses EnhancedStudentModel architecture."""
        try:
            # Try to import the actual student model from KnowledgeDistillation
            import sys
            kd_path = Path("D:/KnowledgeDistillation/src")
            if kd_path.exists() and str(kd_path) not in sys.path:
                sys.path.insert(0, str(kd_path))
            
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            class_names = checkpoint.get('class_names', self.class_names)
            
            # Determine num_classes from main classifier in state_dict (not from class_names which may include junk)
            main_classifier_key = 'classifier.6.weight'  # Main classifier layer
            if main_classifier_key in state_dict:
                num_classes = state_dict[main_classifier_key].shape[0]
            else:
                num_classes = len(class_names) if class_names else self.num_classes
            
            # Update class names if available in checkpoint
            if class_names:
                self.class_names = class_names
                self.num_classes = num_classes
            
            # Try to use the actual EnhancedStudentModel
            try:
                from enhanced_student_model import EnhancedStudentModel
                
                # Infer base_channels from state dict
                stem_channels = state_dict.get('stem.0.weight', None)
                base_channels = stem_channels.shape[0] if stem_channels is not None else 48
                
                model = EnhancedStudentModel(
                    num_classes=num_classes,
                    input_channels=3,
                    input_size=256,
                    base_channels=base_channels
                )
                
                # Filter state dict to skip mismatched aux_classifiers (not needed for inference)
                model_state = model.state_dict()
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    if key in model_state:
                        if model_state[key].shape == value.shape:
                            filtered_state_dict[key] = value
                        else:
                            logger.debug(f"Skipping mismatched layer: {key}")
                    else:
                        filtered_state_dict[key] = value
                
                # Load weights (allow partial loading for any remaining mismatches)
                model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"Successfully created EnhancedStudentModel with {num_classes} classes")
                return model
                return model
                
            except ImportError:
                logger.warning("EnhancedStudentModel not found, trying StudentCNN")
                
                try:
                    from student_model import StudentCNN
                    
                    model = StudentCNN(num_classes=num_classes)
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Successfully created StudentCNN with {num_classes} classes")
                    return model
                    
                except ImportError:
                    logger.warning("StudentCNN not found, falling back to MobileNetV3")
                    import torchvision.models as models
                    model = models.mobilenet_v3_small(weights=None)
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
                    model.load_state_dict(state_dict, strict=False)
                    logger.info("Fallback: Created MobileNetV3-Small model from state dict")
                    return model
            
        except Exception as e:
            logger.error(f"Could not create model from state dict: {e}")
            return checkpoint  # Return as-is
    
    def _bytes_to_numpy(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert image bytes to numpy array.
        
        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)
            
        Returns:
            Numpy array in RGB format (H, W, C)
        """
        from PIL import Image
        import io
        
        # Decode image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to expected input size (256x256)
        img = img.resize((256, 256), Image.BILINEAR)
        
        # Convert to numpy array
        return np.array(img)
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB
            
        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # Convert to tensor
        if image.dtype == np.uint8:
            tensor = torch.from_numpy(image).float() / 255.0
        else:
            tensor = torch.from_numpy(image).float()
        
        # Handle grayscale
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        elif tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1).unsqueeze(0).repeat(3, 1, 1)
        elif tensor.shape[-1] == 3:
            # HWC to CHW
            tensor = tensor.permute(2, 0, 1)
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        # Half precision
        if self.half_precision:
            tensor = tensor.half()
        
        return tensor
    
    def predict(
        self,
        image: Union[np.ndarray, bytes],
        return_features: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on an image.
        
        Args:
            image: Input image as numpy array or bytes
            return_features: Whether to return feature vectors
            
        Returns:
            Dictionary with predictions
        """
        if self._is_state_dict:
            raise RuntimeError("Cannot run inference on state dict. Need full model.")
        
        start_time = time.time()
        
        # Convert bytes to numpy array if needed
        if isinstance(image, bytes):
            image = self._bytes_to_numpy(image)
        
        # Preprocess
        tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
                features = outputs[1] if len(outputs) > 1 and return_features else None
            elif isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("output", list(outputs.values())[0]))
                features = outputs.get("features") if return_features else None
            else:
                logits = outputs
                features = None
            
            # Get probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get prediction
            confidence, predicted = torch.max(probabilities, 1)
        
        inference_time = (time.time() - start_time) * 1000
        
        predicted_idx = int(predicted.item())
        class_name = self.class_names[predicted_idx] if predicted_idx < len(self.class_names) else f"class_{predicted_idx}"
        
        result = {
            "logits": logits.cpu().numpy(),
            "probabilities": probabilities.cpu().numpy().flatten(),
            "predicted_class": predicted_idx,
            "class_name": class_name,
            "confidence": float(confidence.item()),
            "inference_time_ms": inference_time,
            # Phase 3 support: include tensor for feature extraction
            "image_tensor": tensor,  # Keep as torch tensor for Phase 3
            "logits_tensor": logits,  # Keep tensor for Phase 3 multi-label
        }
        
        if features is not None:
            result["features"] = features.cpu().numpy()
        
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
        if self._is_state_dict:
            raise RuntimeError("Cannot run inference on state dict. Need full model.")
        
        start_time = time.time()
        
        # Preprocess all images
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(batch)
            
            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("output", list(outputs.values())[0]))
            else:
                logits = outputs
            
            probabilities = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Build results
        results = []
        probs_np = probabilities.cpu().numpy()
        logits_np = logits.cpu().numpy()
        
        for i in range(len(images)):
            results.append({
                "logits": logits_np[i],
                "probabilities": probs_np[i],
                "predicted_class": int(predictions[i].item()),
                "confidence": float(confidences[i].item()),
                "inference_time_ms": inference_time / len(images),
            })
        
        return results
