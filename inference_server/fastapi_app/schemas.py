"""
Pydantic Schemas
================
Request and response schemas for the FastAPI application.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for when pydantic is not installed
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        
        def dict(self):
            return self.__dict__.copy()
        
        def model_dump(self):
            return self.dict()
    
    def Field(default=None, **kwargs):
        return default


# Base Response
class BaseResponse(BaseModel):
    """Base response model."""
    status: str = Field(..., description="Response status")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: Optional[str] = Field(None, description="Response timestamp")


# Error Response
class ErrorDetail(BaseModel):
    """Error detail model."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class ErrorResponse(BaseResponse):
    """Error response model."""
    error: ErrorDetail


# Prediction Models
class PredictionResult(BaseModel):
    """Single prediction result."""
    class_name: str = Field(..., alias="class", description="Predicted class name")
    class_id: int = Field(..., description="Predicted class ID")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    all_probabilities: Optional[Dict[str, float]] = Field(None, description="All class probabilities")
    
    class Config:
        populate_by_name = True


class InferenceInfo(BaseModel):
    """Inference metadata."""
    model_format: str = Field(..., description="Model format used")
    device: str = Field(..., description="Compute device")
    time_ms: float = Field(..., description="Inference time in milliseconds")


class ValidationInfo(BaseModel):
    """Image validation info."""
    passed: bool = Field(..., description="Whether validation passed")
    relevance_score: float = Field(..., ge=0, le=1, description="Content relevance score")
    quality_score: Optional[float] = Field(None, description="Image quality score")


class PredictionResponse(BaseResponse):
    """Single prediction response."""
    prediction: PredictionResult
    inference: InferenceInfo
    validation: Optional[ValidationInfo] = None
    feedback_id: Optional[str] = Field(None, description="ID to submit feedback on this prediction")


# Batch Prediction Models
class BatchResultItem(BaseModel):
    """Single item in batch results."""
    index: int = Field(..., description="Index in batch")
    filename: Optional[str] = Field(None, description="Original filename")
    status: str = Field(..., description="Item status")
    prediction: Optional[PredictionResult] = None
    error: Optional[str] = Field(None, description="Error message if failed")
    inference_time_ms: Optional[float] = Field(None, description="Inference time")


class BatchSummary(BaseModel):
    """Batch processing summary."""
    total: int = Field(..., description="Total images processed")
    successful: int = Field(..., description="Successfully processed")
    failed: int = Field(..., description="Failed to process")
    total_time_ms: float = Field(..., description="Total processing time")
    avg_time_ms: float = Field(..., description="Average time per image")


class BatchPredictionResponse(BaseResponse):
    """Batch prediction response."""
    summary: BatchSummary
    results: List[BatchResultItem]


# Health Check
class ModelInfo(BaseModel):
    """Model information."""
    loaded: bool = Field(..., description="Whether model is loaded")
    info: Optional[Dict[str, Any]] = Field(None, description="Model details")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    model: ModelInfo


# Classes
class ClassInfo(BaseModel):
    """Class information."""
    id: int = Field(..., description="Class ID")
    name: str = Field(..., description="Class name")


class ClassesResponse(BaseResponse):
    """Classes list response."""
    num_classes: int = Field(..., description="Number of classes")
    classes: List[ClassInfo]


# Models Info
class ExposedModelInfo(BaseModel):
    """Publicly exposed model info."""
    name: str = Field(..., description="Model name")
    description: Optional[str] = Field(None, description="Model description")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    formats: List[str] = Field(..., description="Available formats")


class ModelsResponse(BaseResponse):
    """Models list response."""
    models: List[ExposedModelInfo]


# Request Bodies
class Base64ImageRequest(BaseModel):
    """Request with base64 encoded image."""
    image_data: str = Field(..., description="Base64 encoded image data")
    include_probabilities: bool = Field(False, description="Include all probabilities")


class RejectionResponse(BaseResponse):
    """Image rejection response."""
    error: ErrorDetail
    suggestion: str = Field(..., description="Suggestion for the user")


# Feedback Models
class FeedbackRequest(BaseModel):
    """User feedback on prediction."""
    feedback_id: str = Field(..., description="Feedback ID from prediction response")
    is_correct: bool = Field(..., description="Was the prediction correct?")
    correct_class: Optional[str] = Field(None, description="Correct class name if incorrect")
    correct_class_id: Optional[int] = Field(None, description="Correct class ID if incorrect")
    user_comment: Optional[str] = Field(None, description="Optional user comment")
    device_info: Optional[str] = Field(None, description="Device information")
    app_version: Optional[str] = Field(None, description="App version")


class FeedbackRecorded(BaseModel):
    """Recorded feedback details."""
    is_correct: bool
    original_prediction: str
    corrected_to: Optional[str] = None


class FeedbackResponse(BaseResponse):
    """Feedback submission response."""
    message: str = Field(..., description="Response message")
    feedback_id: str = Field(..., description="Feedback ID")
    recorded: FeedbackRecorded


class FeedbackStatsResponse(BaseResponse):
    """Feedback statistics response."""
    total_predictions: int
    feedback_received: int
    correct_predictions: int
    incorrect_predictions: int
    accuracy_from_feedback: Optional[float]
    pending_feedbacks: int
    corrections_by_class: Dict[str, int]
