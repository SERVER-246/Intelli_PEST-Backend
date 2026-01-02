"""
Image Preprocessing
===================
Image preprocessing utilities for inference.
"""

import logging
from typing import Tuple, Optional, Union
from pathlib import Path
import io

logger = logging.getLogger(__name__)

# Lazy imports
PIL_AVAILABLE = False
CV2_AVAILABLE = False
NUMPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    pass

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    pass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass


class ImagePreprocessor:
    """
    Image preprocessing for model inference.
    
    Handles:
    - Image loading from bytes/file
    - Resizing to target dimensions
    - Color space conversion
    - Normalization
    """
    
    # ImageNet normalization parameters
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        interpolation: str = "bilinear",
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (H, W)
            mean: Normalization mean (RGB)
            std: Normalization std (RGB)
            interpolation: Resize interpolation method
        """
        self.target_size = target_size
        self.mean = mean or tuple(self.IMAGENET_MEAN)
        self.std = std or tuple(self.IMAGENET_STD)
        self.interpolation = interpolation
        
        # Select backend
        if CV2_AVAILABLE and NUMPY_AVAILABLE:
            self._backend = "cv2"
        elif PIL_AVAILABLE and NUMPY_AVAILABLE:
            self._backend = "pil"
        else:
            raise ImportError("Either OpenCV or PIL is required for image preprocessing")
        
        logger.debug(f"Using {self._backend} backend for preprocessing")
    
    def load_image(self, source: Union[bytes, str, Path]) -> np.ndarray:
        """
        Load image from bytes or file path.
        
        Args:
            source: Image bytes or file path
            
        Returns:
            Image as numpy array (H, W, C) in RGB
        """
        if isinstance(source, bytes):
            return self._load_from_bytes(source)
        else:
            return self._load_from_file(Path(source))
    
    def _load_from_bytes(self, data: bytes) -> np.ndarray:
        """Load image from bytes."""
        if self._backend == "cv2":
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = Image.open(io.BytesIO(data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = np.array(img)
        
        return img
    
    def _load_from_file(self, path: Path) -> np.ndarray:
        """Load image from file."""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        if self._backend == "cv2":
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = np.array(img)
        
        return img
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        if h == target_h and w == target_w:
            return image
        
        if self._backend == "cv2":
            interp_map = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "lanczos": cv2.INTER_LANCZOS4,
            }
            interp = interp_map.get(self.interpolation, cv2.INTER_LINEAR)
            resized = cv2.resize(image, (target_w, target_h), interpolation=interp)
        else:
            pil_img = Image.fromarray(image)
            interp_map = {
                "nearest": Image.NEAREST,
                "bilinear": Image.BILINEAR,
                "bicubic": Image.BICUBIC,
                "lanczos": Image.LANCZOS,
            }
            interp = interp_map.get(self.interpolation, Image.BILINEAR)
            pil_img = pil_img.resize((target_w, target_h), interp)
            resized = np.array(pil_img)
        
        return resized
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image.
        
        Args:
            image: Input image (H, W, C), 0-255 uint8 or 0-1 float
            
        Returns:
            Normalized image as float32
        """
        # Convert to float if needed
        if image.dtype == np.uint8:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)
        
        # Apply normalization
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        
        normalized = (img - mean) / std
        
        return normalized
    
    def preprocess(
        self,
        source: Union[bytes, str, Path, np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            source: Image source (bytes, path, or numpy array)
            normalize: Whether to apply normalization
            
        Returns:
            Preprocessed image (H, W, C)
        """
        # Load if needed
        if isinstance(source, np.ndarray):
            image = source.copy()
        else:
            image = self.load_image(source)
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]  # Remove alpha
        
        # Resize
        image = self.resize(image)
        
        # Normalize
        if normalize:
            image = self.normalize(image)
        
        return image
    
    def preprocess_batch(
        self,
        sources: list,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Preprocess multiple images.
        
        Args:
            sources: List of image sources
            normalize: Whether to apply normalization
            
        Returns:
            Batch of preprocessed images (N, H, W, C)
        """
        images = [self.preprocess(src, normalize) for src in sources]
        return np.stack(images, axis=0)


# Global preprocessor instance
_preprocessor: Optional[ImagePreprocessor] = None


def get_preprocessor(**kwargs) -> ImagePreprocessor:
    """Get or create the global preprocessor."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = ImagePreprocessor(**kwargs)
    return _preprocessor


def preprocess_image(
    source: Union[bytes, str, Path, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess an image for inference.
    
    Args:
        source: Image source
        target_size: Target size (H, W)
        normalize: Whether to normalize
        
    Returns:
        Preprocessed image
    """
    preprocessor = get_preprocessor(target_size=target_size)
    return preprocessor.preprocess(source, normalize)


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Resize an image.
    
    Args:
        image: Input image
        target_size: Target size (H, W)
        
    Returns:
        Resized image
    """
    preprocessor = get_preprocessor(target_size=target_size)
    return preprocessor.resize(image)
