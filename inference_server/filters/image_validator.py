"""
Image Validator (Layer 2)
=========================
Validates image integrity and properties:
- Image decode test
- Dimension validation
- Aspect ratio check
- Color mode validation
"""

import logging
from typing import Tuple, Optional
from dataclasses import dataclass
import io

logger = logging.getLogger(__name__)

# Lazy imports for image libraries
PIL_AVAILABLE = False
CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("PIL not available, some image validation features disabled")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available, some image validation features disabled")


@dataclass
class ImageValidationResult:
    """Result of image validation."""
    valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    width: int = 0
    height: int = 0
    channels: int = 0
    mode: Optional[str] = None
    aspect_ratio: float = 0.0
    quality_score: float = 1.0


class ImageValidator:
    """
    Layer 2: Image integrity validation.
    
    Validates that the file is a valid image with acceptable properties.
    """
    
    def __init__(
        self,
        min_dimension: int = 64,
        max_dimension: int = 4096,
        min_aspect_ratio: float = 0.25,
        max_aspect_ratio: float = 4.0,
        require_rgb: bool = True,
        check_corruption: bool = True,
    ):
        """
        Initialize image validator.
        
        Args:
            min_dimension: Minimum width/height in pixels
            max_dimension: Maximum width/height in pixels
            min_aspect_ratio: Minimum aspect ratio (width/height)
            max_aspect_ratio: Maximum aspect ratio
            require_rgb: Require RGB/3-channel images
            check_corruption: Check for image corruption
        """
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.require_rgb = require_rgb
        self.check_corruption = check_corruption
    
    def validate(self, image_data: bytes) -> ImageValidationResult:
        """
        Validate image data.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            ImageValidationResult with validation status
        """
        # Try PIL first (more comprehensive)
        if PIL_AVAILABLE:
            return self._validate_with_pil(image_data)
        elif CV2_AVAILABLE:
            return self._validate_with_cv2(image_data)
        else:
            logger.error("No image library available for validation")
            return ImageValidationResult(
                valid=False,
                error_code="NO_IMAGE_LIBRARY",
                error_message="Image processing library not available",
            )
    
    def _validate_with_pil(self, image_data: bytes) -> ImageValidationResult:
        """Validate using PIL/Pillow."""
        try:
            # Try to open image
            img = Image.open(io.BytesIO(image_data))
            
            # Verify image integrity (fully decode)
            if self.check_corruption:
                try:
                    img.verify()
                    # Re-open after verify (verify() makes image unusable)
                    img = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    return ImageValidationResult(
                        valid=False,
                        error_code="CORRUPT_IMAGE",
                        error_message=f"Image appears to be corrupted: {str(e)}",
                    )
            
            # Get dimensions
            width, height = img.size
            mode = img.mode
            
            # Determine channel count
            channels = len(img.getbands())
            
            # Calculate aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            
            # Check dimensions
            if width < self.min_dimension or height < self.min_dimension:
                return ImageValidationResult(
                    valid=False,
                    error_code="IMAGE_TOO_SMALL",
                    error_message=f"Image dimensions ({width}x{height}) below minimum ({self.min_dimension}px)",
                    width=width,
                    height=height,
                    channels=channels,
                    mode=mode,
                    aspect_ratio=aspect_ratio,
                )
            
            if width > self.max_dimension or height > self.max_dimension:
                return ImageValidationResult(
                    valid=False,
                    error_code="IMAGE_TOO_LARGE",
                    error_message=f"Image dimensions ({width}x{height}) exceed maximum ({self.max_dimension}px)",
                    width=width,
                    height=height,
                    channels=channels,
                    mode=mode,
                    aspect_ratio=aspect_ratio,
                )
            
            # Check aspect ratio
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                return ImageValidationResult(
                    valid=False,
                    error_code="INVALID_ASPECT_RATIO",
                    error_message=f"Aspect ratio ({aspect_ratio:.2f}) outside allowed range ({self.min_aspect_ratio}-{self.max_aspect_ratio})",
                    width=width,
                    height=height,
                    channels=channels,
                    mode=mode,
                    aspect_ratio=aspect_ratio,
                )
            
            # Check color mode
            if self.require_rgb and mode not in ["RGB", "RGBA", "L"]:
                return ImageValidationResult(
                    valid=False,
                    error_code="INVALID_COLOR_MODE",
                    error_message=f"Color mode '{mode}' not supported. Expected RGB/RGBA/Grayscale.",
                    width=width,
                    height=height,
                    channels=channels,
                    mode=mode,
                    aspect_ratio=aspect_ratio,
                )
            
            # Calculate quality score based on resolution
            quality_score = min(1.0, (width * height) / (224 * 224))
            
            # All checks passed
            return ImageValidationResult(
                valid=True,
                width=width,
                height=height,
                channels=channels,
                mode=mode,
                aspect_ratio=aspect_ratio,
                quality_score=quality_score,
            )
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return ImageValidationResult(
                valid=False,
                error_code="DECODE_ERROR",
                error_message=f"Failed to decode image: {str(e)}",
            )
    
    def _validate_with_cv2(self, image_data: bytes) -> ImageValidationResult:
        """Validate using OpenCV."""
        try:
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                return ImageValidationResult(
                    valid=False,
                    error_code="DECODE_ERROR",
                    error_message="Failed to decode image with OpenCV",
                )
            
            # Get dimensions
            if len(img.shape) == 2:
                height, width = img.shape
                channels = 1
                mode = "L"  # Grayscale
            else:
                height, width, channels = img.shape
                mode = "RGBA" if channels == 4 else "RGB"
            
            aspect_ratio = width / height if height > 0 else 0
            
            # Check dimensions
            if width < self.min_dimension or height < self.min_dimension:
                return ImageValidationResult(
                    valid=False,
                    error_code="IMAGE_TOO_SMALL",
                    error_message=f"Image dimensions ({width}x{height}) below minimum",
                    width=width,
                    height=height,
                    channels=channels,
                    mode=mode,
                    aspect_ratio=aspect_ratio,
                )
            
            if width > self.max_dimension or height > self.max_dimension:
                return ImageValidationResult(
                    valid=False,
                    error_code="IMAGE_TOO_LARGE",
                    error_message=f"Image dimensions ({width}x{height}) exceed maximum",
                    width=width,
                    height=height,
                    channels=channels,
                    mode=mode,
                    aspect_ratio=aspect_ratio,
                )
            
            # Check aspect ratio
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                return ImageValidationResult(
                    valid=False,
                    error_code="INVALID_ASPECT_RATIO",
                    error_message=f"Aspect ratio ({aspect_ratio:.2f}) outside allowed range",
                    width=width,
                    height=height,
                    channels=channels,
                    mode=mode,
                    aspect_ratio=aspect_ratio,
                )
            
            quality_score = min(1.0, (width * height) / (224 * 224))
            
            return ImageValidationResult(
                valid=True,
                width=width,
                height=height,
                channels=channels,
                mode=mode,
                aspect_ratio=aspect_ratio,
                quality_score=quality_score,
            )
            
        except Exception as e:
            logger.error(f"OpenCV validation failed: {e}")
            return ImageValidationResult(
                valid=False,
                error_code="DECODE_ERROR",
                error_message=f"Failed to process image: {str(e)}",
            )


# Global validator instance
_validator: Optional[ImageValidator] = None


def get_image_validator(**kwargs) -> ImageValidator:
    """Get or create the global image validator."""
    global _validator
    if _validator is None:
        _validator = ImageValidator(**kwargs)
    return _validator


def validate_image(image_data: bytes) -> ImageValidationResult:
    """Validate an image using the global validator."""
    return get_image_validator().validate(image_data)
