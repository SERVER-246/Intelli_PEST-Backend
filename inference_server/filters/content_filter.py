"""
Content Relevance Filter (Layer 3) - Simplified
================================================
Simplified filter that only blocks obvious non-plant images:
- Face detection (selfies, portraits)
- Synthetic image detection (solid colors, screenshots)

The model will handle classification, and users can report "junk" 
images via feedback for continuous learning.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

# Lazy imports
CV2_AVAILABLE = False
NUMPY_AVAILABLE = False

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


@dataclass
class ContentFilterResult:
    """Result of content relevance filtering."""
    relevant: bool
    relevance_score: float
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    detected_category: Optional[str] = None
    analysis_details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.analysis_details is None:
            self.analysis_details = {}


class ContentFilter:
    """
    Layer 3: Simplified content relevance filtering.
    
    Only blocks obvious non-plant images:
    1. Faces (selfies, portraits)
    2. Pure synthetic images (solid colors, perfect gradients)
    
    All other images pass through to the model.
    Users can report irrelevant images via the "junk" feedback option.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize content filter.
        
        Args:
            enabled: Whether to enable filtering (can be disabled)
        """
        self.enabled = enabled
        self._face_cascade = None
    
    def filter(self, image_data: bytes) -> ContentFilterResult:
        """
        Filter image for content relevance.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            ContentFilterResult with relevance assessment
        """
        # If disabled, pass everything
        if not self.enabled:
            return ContentFilterResult(
                relevant=True,
                relevance_score=1.0,
                detected_category="not_filtered",
                analysis_details={"filter_enabled": False},
            )
        
        if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
            logger.warning("Content filtering libraries not available, skipping check")
            return ContentFilterResult(
                relevant=True,
                relevance_score=0.5,
                detected_category="unknown",
                analysis_details={"warning": "Content filtering unavailable"},
            )
        
        try:
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return ContentFilterResult(
                    relevant=False,
                    relevance_score=0.0,
                    error_code="DECODE_ERROR",
                    error_message="Could not decode image for content analysis",
                )
            
            # Resize for faster processing
            max_dim = 512
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            analysis = {}
            
            # 1. Face detection - reject selfies/portraits
            has_face, num_faces = self._detect_faces(img)
            analysis["has_face"] = has_face
            analysis["num_faces"] = num_faces
            
            if has_face:
                return ContentFilterResult(
                    relevant=False,
                    relevance_score=0.0,
                    error_code="HUMAN_DETECTED",
                    error_message="Image appears to contain a person/face. Please upload an image of a plant or pest damage.",
                    detected_category="human_person",
                    analysis_details=analysis,
                )
            
            # 2. Check for pure synthetic images (solid colors, etc.)
            is_synthetic, synthetic_reason = self._check_synthetic(img)
            analysis["is_synthetic"] = is_synthetic
            analysis["synthetic_reason"] = synthetic_reason
            
            if is_synthetic:
                return ContentFilterResult(
                    relevant=False,
                    relevance_score=0.1,
                    error_code="SYNTHETIC_IMAGE",
                    error_message=f"Image appears to be synthetic ({synthetic_reason}). Please capture a real photo of the plant.",
                    detected_category="synthetic_image",
                    analysis_details=analysis,
                )
            
            # All other images pass through
            return ContentFilterResult(
                relevant=True,
                relevance_score=0.8,
                detected_category="potential_plant",
                analysis_details=analysis,
            )
            
        except Exception as e:
            logger.error(f"Content filtering error: {e}")
            # On error, allow through
            return ContentFilterResult(
                relevant=True,
                relevance_score=0.5,
                detected_category="error",
                analysis_details={"error": str(e)},
            )
    
    def _detect_faces(self, img: np.ndarray) -> tuple:
        """
        Detect faces using Haar cascade.
        
        Returns:
            Tuple of (has_face: bool, num_faces: int)
        """
        try:
            if self._face_cascade is None:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            num_faces = len(faces)
            return num_faces > 0, num_faces
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return False, 0
    
    def _check_synthetic(self, img: np.ndarray) -> tuple:
        """
        Check if image is obviously synthetic (solid color, screenshot, etc.)
        
        Returns:
            Tuple of (is_synthetic: bool, reason: str)
        """
        try:
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Check for solid/near-solid color
            std_dev = np.std(gray)
            if std_dev < 10:
                return True, "solid_color"
            
            # Check for very few unique colors (synthetic graphics)
            unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
            total_pixels = h * w
            color_ratio = unique_colors / total_pixels
            
            if unique_colors < 100 and color_ratio < 0.001:
                return True, "limited_colors"
            
            # Check for pure gradients (artificial)
            # Compute Laplacian variance (texture measure)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            if variance < 50 and std_dev < 30:
                return True, "no_texture"
            
            return False, None
            
        except Exception as e:
            logger.warning(f"Synthetic check failed: {e}")
            return False, None


# Global filter instance
_filter: Optional[ContentFilter] = None


def get_content_filter(enabled: bool = True) -> ContentFilter:
    """Get or create the global content filter."""
    global _filter
    if _filter is None:
        _filter = ContentFilter(enabled=enabled)
    return _filter


def check_content_relevance(image_data: bytes) -> ContentFilterResult:
    """Check content relevance using the global filter."""
    return get_content_filter().filter(image_data)
