"""
Content Relevance Filter (Layer 3)
==================================
Filters out irrelevant/junk images that are not related to pest detection:
- Domain classification (agricultural/plant images)
- Color histogram analysis (vegetation patterns)
- Edge density analysis
- Texture analysis for natural images
"""

import logging
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import io
import math

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
    Layer 3: Content relevance filtering.
    
    Uses multiple heuristics to determine if an image is likely
    related to sugarcane/agricultural pest detection.
    
    Features analyzed:
    1. Color distribution (green/brown for vegetation)
    2. Edge density (natural textures)
    3. Color entropy (natural variation)
    4. Vegetation index estimation
    """
    
    # Color ranges for vegetation (in HSV)
    # Green vegetation: H=35-85, S=25-255, V=25-255
    GREEN_LOWER = (35, 25, 25)
    GREEN_UPPER = (85, 255, 255)
    
    # Brown/dead vegetation: H=10-30, S=25-200, V=25-200
    BROWN_LOWER = (10, 25, 25)
    BROWN_UPPER = (30, 200, 200)
    
    # Yellow (stressed vegetation): H=20-35, S=25-255, V=100-255
    YELLOW_LOWER = (20, 25, 100)
    YELLOW_UPPER = (35, 255, 255)
    
    def __init__(
        self,
        relevance_threshold: float = 0.50,
        min_vegetation_ratio: float = 0.15,
        min_natural_score: float = 0.30,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize content filter.
        
        Args:
            relevance_threshold: Minimum score to consider relevant
            min_vegetation_ratio: Minimum vegetation color ratio
            min_natural_score: Minimum natural image score
            weights: Custom weights for different features
        """
        self.relevance_threshold = relevance_threshold
        self.min_vegetation_ratio = min_vegetation_ratio
        self.min_natural_score = min_natural_score
        
        # Default weights for combining features
        self.weights = weights or {
            "vegetation_ratio": 0.35,
            "edge_density": 0.20,
            "color_entropy": 0.15,
            "texture_score": 0.15,
            "aspect_naturalness": 0.15,
        }
    
    def filter(self, image_data: bytes) -> ContentFilterResult:
        """
        Filter image for content relevance.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            ContentFilterResult with relevance assessment
        """
        if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
            # If libraries not available, pass through with warning
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
            
            # Resize for faster processing if needed
            max_dim = 512
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            # Run analysis
            analysis = {}
            
            # 1. Vegetation color analysis
            veg_ratio, veg_details = self._analyze_vegetation_colors(img)
            analysis["vegetation_ratio"] = veg_ratio
            analysis["vegetation_details"] = veg_details
            
            # 2. Edge density analysis
            edge_density = self._analyze_edge_density(img)
            analysis["edge_density"] = edge_density
            
            # 3. Color entropy (natural variation)
            color_entropy = self._analyze_color_entropy(img)
            analysis["color_entropy"] = color_entropy
            
            # 4. Texture analysis
            texture_score = self._analyze_texture(img)
            analysis["texture_score"] = texture_score
            
            # 5. Aspect naturalness (not synthetic/screen capture)
            naturalness = self._check_naturalness(img)
            analysis["aspect_naturalness"] = naturalness
            
            # Calculate weighted score
            relevance_score = (
                self.weights["vegetation_ratio"] * min(veg_ratio / 0.3, 1.0) +
                self.weights["edge_density"] * min(edge_density / 0.15, 1.0) +
                self.weights["color_entropy"] * min(color_entropy / 5.0, 1.0) +
                self.weights["texture_score"] * texture_score +
                self.weights["aspect_naturalness"] * naturalness
            )
            
            # Clamp to 0-1
            relevance_score = max(0.0, min(1.0, relevance_score))
            
            # Determine category
            category = self._determine_category(analysis, relevance_score)
            
            # Check if relevant
            is_relevant = (
                relevance_score >= self.relevance_threshold and
                veg_ratio >= self.min_vegetation_ratio * 0.5  # Some tolerance
            )
            
            return ContentFilterResult(
                relevant=is_relevant,
                relevance_score=relevance_score,
                detected_category=category,
                analysis_details=analysis,
                error_code=None if is_relevant else "CONTENT_IRRELEVANT",
                error_message=None if is_relevant else f"Image does not appear to be agricultural/plant-related (score: {relevance_score:.2f})",
            )
            
        except Exception as e:
            logger.error(f"Content filtering error: {e}")
            # On error, allow through but flag it
            return ContentFilterResult(
                relevant=True,
                relevance_score=0.5,
                detected_category="error",
                analysis_details={"error": str(e)},
            )
    
    def _analyze_vegetation_colors(self, img: np.ndarray) -> Tuple[float, Dict]:
        """Analyze vegetation-related colors in image."""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create masks for different vegetation colors
        green_mask = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)
        brown_mask = cv2.inRange(hsv, self.BROWN_LOWER, self.BROWN_UPPER)
        yellow_mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
        
        # Calculate ratios
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = np.sum(green_mask > 0) / total_pixels
        brown_ratio = np.sum(brown_mask > 0) / total_pixels
        yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
        
        # Combined vegetation ratio
        vegetation_ratio = green_ratio + brown_ratio * 0.7 + yellow_ratio * 0.5
        
        details = {
            "green_ratio": float(green_ratio),
            "brown_ratio": float(brown_ratio),
            "yellow_ratio": float(yellow_ratio),
            "combined": float(vegetation_ratio),
        }
        
        return float(vegetation_ratio), details
    
    def _analyze_edge_density(self, img: np.ndarray) -> float:
        """Analyze edge density (natural images have moderate edge density)."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return float(edge_density)
    
    def _analyze_color_entropy(self, img: np.ndarray) -> float:
        """Calculate color entropy (natural images have higher entropy)."""
        # Convert to grayscale for histogram
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Calculate entropy
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        
        return float(entropy)
    
    def _analyze_texture(self, img: np.ndarray) -> float:
        """Analyze texture using Laplacian variance."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance (measure of texture/sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range (typical values 0-5000)
        texture_score = min(1.0, variance / 2000)
        
        return float(texture_score)
    
    def _check_naturalness(self, img: np.ndarray) -> float:
        """
        Check if image appears natural vs synthetic/screen capture.
        
        Synthetic images often have:
        - Very uniform colors
        - Sharp artificial edges
        - Perfect gradients
        """
        h, w = img.shape[:2]
        score = 1.0
        
        # Check for too-uniform colors (screenshots, solid backgrounds)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        if std_dev < 20:  # Very uniform
            score *= 0.5
        
        # Check for perfect horizontal/vertical lines (UI elements)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            h_lines = 0
            v_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.atan2(y2-y1, x2-x1) * 180 / math.pi)
                if angle < 5 or angle > 175:  # Horizontal
                    h_lines += 1
                elif 85 < angle < 95:  # Vertical
                    v_lines += 1
            
            # Many perfect lines suggest synthetic image
            if h_lines + v_lines > 10:
                score *= 0.7
        
        # Check for unnatural color distributions
        channels = cv2.split(img)
        for channel in channels:
            unique_colors = len(np.unique(channel))
            if unique_colors < 50:  # Very few unique values
                score *= 0.9
        
        return score
    
    def _determine_category(self, analysis: Dict, score: float) -> str:
        """Determine image category based on analysis."""
        veg_ratio = analysis.get("vegetation_ratio", 0)
        edge_density = analysis.get("edge_density", 0)
        naturalness = analysis.get("aspect_naturalness", 0)
        
        if veg_ratio > 0.3 and naturalness > 0.7:
            return "plant_vegetation"
        elif veg_ratio > 0.15 and edge_density > 0.1:
            return "agricultural_scene"
        elif naturalness < 0.5:
            return "synthetic_screenshot"
        elif veg_ratio < 0.05:
            return "non_vegetation"
        elif score >= self.relevance_threshold:
            return "possibly_relevant"
        else:
            return "irrelevant"


# Global filter instance
_filter: Optional[ContentFilter] = None


def get_content_filter(**kwargs) -> ContentFilter:
    """Get or create the global content filter."""
    global _filter
    if _filter is None:
        _filter = ContentFilter(**kwargs)
    return _filter


def check_content_relevance(image_data: bytes) -> ContentFilterResult:
    """Check content relevance using the global filter."""
    return get_content_filter().filter(image_data)
