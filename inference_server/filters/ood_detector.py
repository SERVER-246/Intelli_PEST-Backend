"""
Out-of-Distribution (OOD) Detector (Layer 4)
============================================
Detects images that are significantly different from the training distribution:
- Feature space distance
- Confidence calibration
- Entropy-based uncertainty
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import io
import math

logger = logging.getLogger(__name__)

# Lazy imports
NUMPY_AVAILABLE = False
TORCH_AVAILABLE = False
PIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    pass


@dataclass
class OODResult:
    """Result of OOD detection."""
    in_distribution: bool
    ood_score: float  # Higher = more likely OOD
    confidence: float  # Model confidence in prediction
    entropy: float  # Prediction entropy
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class OODDetector:
    """
    Layer 4: Out-of-Distribution Detection.
    
    Uses multiple methods to detect if an image is likely outside
    the model's training distribution:
    
    1. Maximum Softmax Probability (MSP) - Low confidence = OOD
    2. Entropy of predictions - High entropy = OOD  
    3. Feature space analysis (if reference features available)
    """
    
    # Reference statistics from training data (approximate for sugarcane pest)
    # These can be calibrated with actual training data
    REFERENCE_STATS = {
        "mean_confidence": 0.85,
        "std_confidence": 0.15,
        "mean_entropy": 0.5,
        "std_entropy": 0.3,
    }
    
    def __init__(
        self,
        ood_threshold: float = 0.70,
        confidence_threshold: float = 0.30,
        entropy_threshold: float = 2.0,
        use_temperature_scaling: bool = True,
        temperature: float = 1.5,
    ):
        """
        Initialize OOD detector.
        
        Args:
            ood_threshold: Threshold for OOD score (above = OOD)
            confidence_threshold: Minimum confidence for in-distribution
            entropy_threshold: Maximum entropy for in-distribution
            use_temperature_scaling: Apply temperature scaling for calibration
            temperature: Temperature for softmax scaling
        """
        self.ood_threshold = ood_threshold
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.use_temperature_scaling = use_temperature_scaling
        self.temperature = temperature
        
        # Feature statistics (can be loaded from training)
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_cov_inv: Optional[np.ndarray] = None
    
    def detect(
        self,
        logits: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> OODResult:
        """
        Detect if input is out-of-distribution.
        
        Args:
            logits: Raw model logits (before softmax)
            probabilities: Softmax probabilities
            features: Feature vectors from model (optional)
            
        Returns:
            OODResult with OOD assessment
        """
        if not NUMPY_AVAILABLE:
            return OODResult(
                in_distribution=True,
                ood_score=0.0,
                confidence=1.0,
                entropy=0.0,
                details={"warning": "NumPy not available"},
            )
        
        details = {}
        
        # Get probabilities
        if probabilities is None and logits is not None:
            if self.use_temperature_scaling:
                scaled_logits = logits / self.temperature
            else:
                scaled_logits = logits
            probabilities = self._softmax(scaled_logits)
        
        if probabilities is None:
            return OODResult(
                in_distribution=True,
                ood_score=0.0,
                confidence=1.0,
                entropy=0.0,
                error_code="NO_PREDICTIONS",
                error_message="No predictions provided for OOD detection",
            )
        
        # Ensure numpy array
        if TORCH_AVAILABLE and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        probabilities = np.array(probabilities).flatten()
        
        # 1. Maximum Softmax Probability (MSP)
        max_prob = float(np.max(probabilities))
        details["max_probability"] = max_prob
        
        # 2. Prediction entropy
        entropy = self._calculate_entropy(probabilities)
        details["entropy"] = entropy
        
        # 3. Energy score (if logits available)
        energy_score = 0.0
        if logits is not None:
            energy_score = self._calculate_energy(logits)
            details["energy_score"] = energy_score
        
        # 4. Mahalanobis distance (if features and reference stats available)
        mahal_distance = 0.0
        if features is not None and self._feature_mean is not None:
            mahal_distance = self._calculate_mahalanobis(features)
            details["mahalanobis_distance"] = mahal_distance
        
        # Calculate combined OOD score
        ood_score = self._calculate_ood_score(
            max_prob=max_prob,
            entropy=entropy,
            energy=energy_score,
            mahal=mahal_distance,
        )
        details["ood_score"] = ood_score
        
        # Determine if in-distribution
        in_distribution = (
            ood_score < self.ood_threshold and
            max_prob >= self.confidence_threshold and
            entropy <= self.entropy_threshold
        )
        
        # Generate message if OOD
        error_code = None
        error_message = None
        if not in_distribution:
            error_code = "OUT_OF_DISTRIBUTION"
            if max_prob < self.confidence_threshold:
                error_message = f"Model confidence too low ({max_prob:.2f}). Image may not be recognizable."
            elif entropy > self.entropy_threshold:
                error_message = f"High prediction uncertainty (entropy: {entropy:.2f}). Image unclear or unfamiliar."
            else:
                error_message = f"Image appears different from training data (OOD score: {ood_score:.2f})."
        
        return OODResult(
            in_distribution=in_distribution,
            ood_score=ood_score,
            confidence=max_prob,
            entropy=entropy,
            error_code=error_code,
            error_message=error_message,
            details=details,
        )
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate entropy of probability distribution."""
        # Add small epsilon to avoid log(0)
        probs = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def _calculate_energy(self, logits: np.ndarray) -> float:
        """
        Calculate energy score.
        Lower energy = more likely in-distribution.
        """
        # Energy = -T * log(sum(exp(logits/T)))
        scaled = logits / self.temperature
        energy = -self.temperature * np.log(np.sum(np.exp(scaled - np.max(scaled))) + 1e-10)
        return float(energy)
    
    def _calculate_mahalanobis(self, features: np.ndarray) -> float:
        """
        Calculate Mahalanobis distance from training distribution.
        """
        if self._feature_mean is None or self._feature_cov_inv is None:
            return 0.0
        
        diff = features - self._feature_mean
        distance = np.sqrt(np.dot(np.dot(diff, self._feature_cov_inv), diff.T))
        return float(distance)
    
    def _calculate_ood_score(
        self,
        max_prob: float,
        entropy: float,
        energy: float = 0.0,
        mahal: float = 0.0,
    ) -> float:
        """
        Calculate combined OOD score.
        
        Higher score = more likely OOD.
        """
        # Normalize components to 0-1 range
        prob_score = 1.0 - max_prob  # Low prob = high OOD score
        
        # Normalize entropy (typical range 0-3 for 11 classes)
        max_entropy = math.log(11)  # Maximum entropy for 11 classes
        entropy_score = min(1.0, entropy / max_entropy)
        
        # Combine scores (weighted average)
        weights = {
            "probability": 0.4,
            "entropy": 0.4,
            "energy": 0.1,
            "mahalanobis": 0.1,
        }
        
        ood_score = (
            weights["probability"] * prob_score +
            weights["entropy"] * entropy_score +
            weights["energy"] * min(1.0, abs(energy) / 10) +
            weights["mahalanobis"] * min(1.0, mahal / 100)
        )
        
        return float(ood_score)
    
    def set_reference_statistics(
        self,
        feature_mean: np.ndarray,
        feature_cov: np.ndarray,
    ):
        """
        Set reference statistics from training data.
        
        Args:
            feature_mean: Mean feature vector from training
            feature_cov: Covariance matrix of training features
        """
        self._feature_mean = feature_mean
        try:
            self._feature_cov_inv = np.linalg.inv(feature_cov)
        except np.linalg.LinAlgError:
            logger.warning("Could not invert covariance matrix, using pseudo-inverse")
            self._feature_cov_inv = np.linalg.pinv(feature_cov)
    
    def load_reference_statistics(self, path: str):
        """Load reference statistics from file."""
        try:
            data = np.load(path)
            self._feature_mean = data["mean"]
            self._feature_cov_inv = data["cov_inv"]
            logger.info(f"Loaded OOD reference statistics from {path}")
        except Exception as e:
            logger.warning(f"Could not load reference statistics: {e}")


# Global detector instance
_detector: Optional[OODDetector] = None


def get_ood_detector(**kwargs) -> OODDetector:
    """Get or create the global OOD detector."""
    global _detector
    if _detector is None:
        _detector = OODDetector(**kwargs)
    return _detector


def check_ood(
    logits: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None,
) -> OODResult:
    """Check for OOD using the global detector."""
    return get_ood_detector().detect(logits, probabilities, features)
