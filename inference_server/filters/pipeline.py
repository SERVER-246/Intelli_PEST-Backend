"""
Validation Pipeline
===================
Combined 4-layer validation pipeline for image uploads.
"""

import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from .file_validator import FileValidator, FileValidationResult
from .image_validator import ImageValidator, ImageValidationResult
from .content_filter import ContentFilter, ContentFilterResult
from .ood_detector import OODDetector, OODResult

logger = logging.getLogger(__name__)


class ValidationLayer(Enum):
    """Validation layers."""
    FILE = "file"
    IMAGE = "image"
    CONTENT = "content"
    OOD = "ood"


@dataclass
class ValidationResult:
    """Combined result of all validation layers."""
    valid: bool
    passed_layers: list
    failed_layer: Optional[ValidationLayer] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    suggestion: Optional[str] = None
    
    # Layer-specific results
    file_result: Optional[FileValidationResult] = None
    image_result: Optional[ImageValidationResult] = None
    content_result: Optional[ContentFilterResult] = None
    ood_result: Optional[OODResult] = None
    
    # Scores
    relevance_score: float = 0.0
    quality_score: float = 0.0
    confidence_score: float = 0.0
    
    # Timing
    validation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "valid": self.valid,
            "passed_layers": self.passed_layers,
            "failed_layer": self.failed_layer.value if self.failed_layer else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "suggestion": self.suggestion,
            "scores": {
                "relevance": self.relevance_score,
                "quality": self.quality_score,
                "confidence": self.confidence_score,
            },
            "validation_time_ms": self.validation_time_ms,
        }


class ValidationPipeline:
    """
    4-Layer validation pipeline for image uploads.
    
    Layers:
    1. File Validation - Size, extension, MIME type, malicious patterns
    2. Image Validation - Decode test, dimensions, aspect ratio, color mode
    3. Content Filter - Relevance to agricultural/pest detection
    4. OOD Detection - Out-of-distribution detection (post-inference)
    
    The first 3 layers run before inference, layer 4 runs after.
    """
    
    # Error code to suggestion mapping
    SUGGESTIONS = {
        "FILE_TOO_SMALL": "Please upload a larger image file.",
        "FILE_TOO_LARGE": "Please reduce the image file size (max 10MB).",
        "INVALID_EXTENSION": "Please upload a JPG, PNG, or WebP image.",
        "UNKNOWN_FILE_TYPE": "Please upload a valid image file.",
        "INVALID_MIME_TYPE": "Please upload a JPG, PNG, or WebP image.",
        "MALICIOUS_CONTENT": "Please upload a clean image file without embedded content.",
        "IMAGE_TOO_SMALL": "Please upload a larger image (minimum 64x64 pixels).",
        "IMAGE_TOO_LARGE": "Please reduce the image dimensions (max 4096x4096 pixels).",
        "INVALID_ASPECT_RATIO": "Please upload an image with a more standard aspect ratio.",
        "INVALID_COLOR_MODE": "Please upload an RGB color image.",
        "CORRUPT_IMAGE": "The image appears corrupted. Please try a different image.",
        "DECODE_ERROR": "Could not process the image. Please try a different format.",
        "CONTENT_IRRELEVANT": "Please upload a clear image of sugarcane plant or suspected pest damage.",
        "OUT_OF_DISTRIBUTION": "The image doesn't appear to match our training data. Please upload a clearer image of the plant/pest.",
    }
    
    def __init__(
        self,
        file_validator: Optional[FileValidator] = None,
        image_validator: Optional[ImageValidator] = None,
        content_filter: Optional[ContentFilter] = None,
        ood_detector: Optional[OODDetector] = None,
        skip_content_filter: bool = False,
        skip_ood_detection: bool = False,
    ):
        """
        Initialize validation pipeline.
        
        Args:
            file_validator: Custom file validator (uses default if None)
            image_validator: Custom image validator (uses default if None)
            content_filter: Custom content filter (uses default if None)
            ood_detector: Custom OOD detector (uses default if None)
            skip_content_filter: Skip content filtering
            skip_ood_detection: Skip OOD detection
        """
        self.file_validator = file_validator or FileValidator()
        self.image_validator = image_validator or ImageValidator()
        self.content_filter = content_filter or ContentFilter()
        self.ood_detector = ood_detector or OODDetector()
        
        self.skip_content_filter = skip_content_filter
        self.skip_ood_detection = skip_ood_detection
    
    def validate_pre_inference(
        self,
        file_data: bytes,
        filename: str,
        content_type: Optional[str] = None,
    ) -> ValidationResult:
        """
        Run validation layers 1-3 (before model inference).
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            content_type: Declared Content-Type
            
        Returns:
            ValidationResult with pre-inference validation status
        """
        start_time = time.time()
        passed_layers = []
        
        # Layer 1: File Validation
        file_result = self.file_validator.validate(file_data, filename, content_type)
        
        if not file_result.valid:
            return ValidationResult(
                valid=False,
                passed_layers=passed_layers,
                failed_layer=ValidationLayer.FILE,
                error_code=file_result.error_code,
                error_message=file_result.error_message,
                suggestion=self.SUGGESTIONS.get(file_result.error_code),
                file_result=file_result,
                validation_time_ms=(time.time() - start_time) * 1000,
            )
        
        passed_layers.append(ValidationLayer.FILE.value)
        
        # Layer 2: Image Validation
        image_result = self.image_validator.validate(file_data)
        
        if not image_result.valid:
            return ValidationResult(
                valid=False,
                passed_layers=passed_layers,
                failed_layer=ValidationLayer.IMAGE,
                error_code=image_result.error_code,
                error_message=image_result.error_message,
                suggestion=self.SUGGESTIONS.get(image_result.error_code),
                file_result=file_result,
                image_result=image_result,
                quality_score=image_result.quality_score,
                validation_time_ms=(time.time() - start_time) * 1000,
            )
        
        passed_layers.append(ValidationLayer.IMAGE.value)
        
        # Layer 3: Content Filter
        content_result = None
        if not self.skip_content_filter:
            content_result = self.content_filter.filter(file_data)
            
            if not content_result.relevant:
                return ValidationResult(
                    valid=False,
                    passed_layers=passed_layers,
                    failed_layer=ValidationLayer.CONTENT,
                    error_code=content_result.error_code,
                    error_message=content_result.error_message,
                    suggestion=self.SUGGESTIONS.get(content_result.error_code),
                    file_result=file_result,
                    image_result=image_result,
                    content_result=content_result,
                    relevance_score=content_result.relevance_score,
                    quality_score=image_result.quality_score,
                    validation_time_ms=(time.time() - start_time) * 1000,
                )
            
            passed_layers.append(ValidationLayer.CONTENT.value)
        
        # All pre-inference validations passed
        return ValidationResult(
            valid=True,
            passed_layers=passed_layers,
            file_result=file_result,
            image_result=image_result,
            content_result=content_result,
            relevance_score=content_result.relevance_score if content_result else 1.0,
            quality_score=image_result.quality_score,
            validation_time_ms=(time.time() - start_time) * 1000,
        )
    
    def validate_post_inference(
        self,
        pre_result: ValidationResult,
        logits=None,
        probabilities=None,
        features=None,
    ) -> ValidationResult:
        """
        Run validation layer 4 (after model inference).
        
        Args:
            pre_result: Result from pre-inference validation
            logits: Model logits
            probabilities: Model probabilities
            features: Feature vectors (optional)
            
        Returns:
            Updated ValidationResult with OOD detection
        """
        if self.skip_ood_detection:
            pre_result.passed_layers.append(ValidationLayer.OOD.value)
            pre_result.confidence_score = float(max(probabilities)) if probabilities is not None else 1.0
            return pre_result
        
        start_time = time.time()
        
        # Layer 4: OOD Detection
        ood_result = self.ood_detector.detect(
            logits=logits,
            probabilities=probabilities,
            features=features,
        )
        
        pre_result.ood_result = ood_result
        pre_result.confidence_score = ood_result.confidence
        
        if not ood_result.in_distribution:
            pre_result.valid = False
            pre_result.failed_layer = ValidationLayer.OOD
            pre_result.error_code = ood_result.error_code
            pre_result.error_message = ood_result.error_message
            pre_result.suggestion = self.SUGGESTIONS.get(ood_result.error_code)
        else:
            pre_result.passed_layers.append(ValidationLayer.OOD.value)
        
        pre_result.validation_time_ms += (time.time() - start_time) * 1000
        
        return pre_result
    
    def validate_full(
        self,
        file_data: bytes,
        filename: str,
        content_type: Optional[str] = None,
        logits=None,
        probabilities=None,
        features=None,
    ) -> ValidationResult:
        """
        Run full validation pipeline (all 4 layers).
        
        This should be called when you have inference results available.
        For pre-inference only, use validate_pre_inference().
        
        Args:
            file_data: Raw file bytes
            filename: Original filename
            content_type: Declared Content-Type
            logits: Model logits (for OOD)
            probabilities: Model probabilities (for OOD)
            features: Feature vectors (for OOD)
            
        Returns:
            ValidationResult with full validation status
        """
        # Run pre-inference validation
        result = self.validate_pre_inference(file_data, filename, content_type)
        
        if not result.valid:
            return result
        
        # Run post-inference validation if we have predictions
        if logits is not None or probabilities is not None:
            result = self.validate_post_inference(
                result,
                logits=logits,
                probabilities=probabilities,
                features=features,
            )
        
        return result


# Global pipeline instance
_pipeline: Optional[ValidationPipeline] = None


def get_validation_pipeline(**kwargs) -> ValidationPipeline:
    """Get or create the global validation pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ValidationPipeline(**kwargs)
    return _pipeline
