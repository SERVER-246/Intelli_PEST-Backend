"""
Pydantic Schemas
================
Request and response schemas for the FastAPI application.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Base Response
class BaseResponse(BaseModel):
    """Base response model."""
    status: str = Field(..., description="Response status")
    request_id: str | None = Field(default=None, description="Request identifier")
    timestamp: str | None = Field(default=None, description="Response timestamp")


# Error Response
class ErrorDetail(BaseModel):
    """Error detail model."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional details")


class ErrorResponse(BaseResponse):
    """Error response model."""
    error: ErrorDetail


# Prediction Models
class PredictionResult(BaseModel):
    """Single prediction result."""
    class_name: str = Field(..., alias="class", description="Predicted class name")
    class_id: int = Field(..., description="Predicted class ID")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    all_probabilities: dict[str, float] | None = Field(default=None, description="All class probabilities")

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
    quality_score: float | None = Field(default=None, description="Image quality score")


# ============================================================
# Phase 3: Experimental Features (Attention Maps, Regions, Multi-Label)
# ============================================================

class Phase3RegionInfo(BaseModel):
    """Region of interest information from Phase 3 analysis."""
    region_id: int = Field(..., description="Region identifier")
    bbox: list[float] | None = Field(default=None, description="Bounding box [x1, y1, x2, y2] normalized 0-1")
    relevance_score: float = Field(..., ge=0, le=1, description="Region relevance score")
    label: str | None = Field(default=None, description="Region label if identified")


class Phase3MultiLabelPrediction(BaseModel):
    """Multi-label prediction from Phase 3."""
    label: str = Field(..., description="Predicted label")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class Phase3AttentionInfo(BaseModel):
    """Attention map information from Phase 3."""
    available: bool = Field(default=False, description="Whether attention map is available")
    map_uri: str | None = Field(default=None, description="URI to attention map image (base64 or URL)")
    method: str | None = Field(default=None, description="Attention extraction method (e.g., 'grad_cam', 'attention')")


class Phase3Response(BaseModel):
    """
    Phase 3 experimental features response.

    These features are EXPERIMENTAL and may change.
    The main prediction should always be trusted over Phase 3 outputs.
    """
    is_experimental: bool = Field(default=True, description="Flag indicating these are experimental features")
    executed: bool = Field(default=False, description="Whether Phase 3 analysis was executed")

    # Attention Maps
    attention: Phase3AttentionInfo | None = Field(default=None, description="Attention map information")

    # Region Analysis
    regions: list[Phase3RegionInfo] | None = Field(default=None, description="Regions of interest")
    top_region_score: float | None = Field(default=None, description="Highest region relevance score")

    # Multi-Label Predictions
    multi_label: list[Phase3MultiLabelPrediction] | None = Field(default=None, description="Multi-label predictions")

    # Metadata
    processing_time_ms: float | None = Field(default=None, description="Phase 3 processing time")
    error: str | None = Field(default=None, description="Error message if Phase 3 failed")


class PredictionResponse(BaseResponse):
    """Single prediction response."""
    prediction: PredictionResult
    inference: InferenceInfo | None = Field(default=None, description="Inference info (omitted in lite mode)")
    validation: ValidationInfo | None = None
    feedback_id: str | None = Field(default=None, description="ID to submit feedback on this prediction")
    # Phase 3: Experimental features (nullable - missing means Phase 3 not enabled/available)
    phase3: Phase3Response | None = Field(default=None, description="Phase 3 experimental features (attention maps, regions, multi-label)")


# Batch Prediction Models
class BatchResultItem(BaseModel):
    """Single item in batch results."""
    index: int = Field(..., description="Index in batch")
    filename: str | None = Field(default=None, description="Original filename")
    status: str = Field(..., description="Item status")
    prediction: PredictionResult | None = None
    error: str | None = Field(default=None, description="Error message if failed")
    inference_time_ms: float | None = Field(default=None, description="Inference time")


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
    results: list[BatchResultItem]


# Health Check
class ModelInfo(BaseModel):
    """Model information."""
    loaded: bool = Field(..., description="Whether model is loaded")
    info: dict[str, Any] | None = Field(default=None, description="Model details")


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
    classes: list[ClassInfo]
    special_categories: list[str] | None = Field(
        default=None,
        description="Special feedback categories like 'junk', 'unrelated' for non-pest images"
    )


# Models Info
class ExposedModelInfo(BaseModel):
    """Publicly exposed model info."""
    name: str = Field(..., description="Model name")
    description: str | None = Field(default=None, description="Model description")
    accuracy: float | None = Field(default=None, description="Model accuracy")
    formats: list[str] = Field(..., description="Available formats")


class ModelsResponse(BaseResponse):
    """Models list response."""
    models: list[ExposedModelInfo]


# Request Bodies
class Base64ImageRequest(BaseModel):
    """Request with base64 encoded image."""
    image_data: str = Field(..., description="Base64 encoded image data")
    include_probabilities: bool = Field(default=False, description="Include all probabilities")


class RejectionResponse(BaseResponse):
    """Image rejection response."""
    error: ErrorDetail
    suggestion: str = Field(..., description="Suggestion for the user")


# Feedback Models
class FeedbackRequest(BaseModel):
    """User feedback on prediction."""
    feedback_id: str = Field(..., description="Feedback ID from prediction response")
    is_correct: bool = Field(..., description="Was the prediction correct?")
    correct_class: str | None = Field(default=None, description="Correct class name if incorrect")
    correct_class_id: int | None = Field(default=None, description="Correct class ID if incorrect")
    user_comment: str | None = Field(default=None, description="Optional user comment")
    device_info: str | None = Field(default=None, description="Device information")
    app_version: str | None = Field(default=None, description="App version")


class FeedbackRecorded(BaseModel):
    """Recorded feedback details."""
    is_correct: bool
    original_prediction: str
    corrected_to: str | None = None


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
    accuracy_from_feedback: float | None
    pending_feedbacks: int
    corrections_by_class: dict[str, int]
    junk_reports: int | None = Field(default=0, description="Number of images reported as junk/unrelated")
    special_categories: dict[str, int] | None = Field(default=None, description="Breakdown by special category type")


# ============================================================
# Connection Quality Tracking (2G/3G/Slow Connection Analytics)
# ============================================================

class ConnectionSample(BaseModel):
    """Single connection quality sample from app."""
    timestamp: int = Field(..., description="Unix timestamp in milliseconds")
    network_type: str = Field(..., description="Network type (wifi, 5g, 4g, 3g, 2g, etc.)")
    quality_level: int = Field(..., ge=0, le=5, description="Quality level 0-5 (offline to excellent)")
    download_speed_kbps: int | None = Field(default=None, description="Measured download speed in Kbps")
    latitude: float | None = Field(default=None, description="GPS latitude")
    longitude: float | None = Field(default=None, description="GPS longitude")


class ConnectionReportRequest(BaseModel):
    """Request to report connection quality samples."""
    device_id: str = Field(..., description="Device identifier")
    user_id: str | None = Field(default=None, description="User identifier")
    app_version: str = Field(..., description="App version string")
    samples: list[ConnectionSample] = Field(..., description="Connection samples to report")


class ConnectionReportResponse(BaseResponse):
    """Response from connection report endpoint."""
    message: str = Field(..., description="Status message")
    samples_received: int = Field(..., description="Number of samples received")


class ConnectionStatsResponse(BaseResponse):
    """Connection quality statistics response."""
    total_samples: int = Field(..., description="Total samples collected")
    samples_by_network_type: dict[str, int] = Field(..., description="Sample count by network type")
    samples_by_quality: dict[str, int] = Field(..., description="Sample count by quality level")
    slow_connection_locations: list[dict[str, Any]] = Field(default_factory=list, description="Locations with slow connections")
    average_speeds_by_type: dict[str, float] = Field(default_factory=dict, description="Average speeds by network type")

