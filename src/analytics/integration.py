"""
ANALYTICS INTEGRATION
=====================
Integrates analytics tracking into the inference server.

This module provides hooks for:
1. Logging predictions during inference
2. Logging corrections when feedback is submitted
3. Model version tracking on startup

Usage:
    # In server startup (run_server.py or main.py):
    from src.analytics.integration import init_analytics
    init_analytics(model_version="v1.0.6")

    # In prediction endpoint (routers.py):
    from src.analytics.integration import log_prediction
    log_prediction(
        prediction_id=feedback_id,
        predicted_class=class_name,
        confidence=confidence,
        image_hash=image_hash
    )

    # In feedback endpoint (routers.py):
    from src.analytics.integration import log_correction
    log_correction(
        prediction_id=feedback_id,
        predicted_class=original_prediction,
        actual_class=corrected_class,
        is_correct=is_correct
    )
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global tracker instance
_tracker = None
_initialized = False


def get_tracker():
    """Get the global correction tracker instance."""
    global _tracker, _initialized

    if not _initialized:
        try:
            from .correction_tracker import CorrectionTracker
            _tracker = CorrectionTracker()
            _initialized = True
            logger.info("Analytics tracker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize analytics tracker: {e}")
            _tracker = None
            _initialized = True  # Mark as initialized to avoid repeated attempts

    return _tracker


def init_analytics(
    model_version: str | None = None,
    data_dir: str | None = None
) -> bool:
    """
    Initialize analytics system with model version.

    Call this on server startup.

    Args:
        model_version: Current model version string (e.g., "v1.0.6")
        data_dir: Directory for analytics data

    Returns:
        True if initialization successful
    """
    global _tracker, _initialized

    try:
        from .correction_tracker import CorrectionTracker

        if data_dir:
            _tracker = CorrectionTracker(data_dir=data_dir)
        else:
            _tracker = CorrectionTracker()

        _initialized = True

        # Set model version if provided
        if model_version:
            _tracker.set_model_version(model_version)
            logger.info(f"Analytics initialized with model version: {model_version}")
        else:
            logger.info("Analytics initialized (no model version set)")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize analytics: {e}")
        import traceback
        traceback.print_exc()
        _tracker = None
        _initialized = True
        return False


def log_prediction(
    prediction_id: str,
    predicted_class: str,
    confidence: float,
    predicted_class_id: int | None = None,
    image_hash: str | None = None,
    user_id: str | None = None,
    attention_map: Any | None = None,
    request_id: str | None = None,
    metadata: dict | None = None
) -> bool:
    """
    Log a prediction for tracking.

    Call this after successful inference in the prediction endpoint.

    Args:
        prediction_id: Unique ID for this prediction (feedback_id)
        predicted_class: Predicted class name
        confidence: Prediction confidence (0-1)
        predicted_class_id: Optional class index
        image_hash: Optional image hash for deduplication
        user_id: Optional user ID
        attention_map: Optional attention map data
        request_id: Optional request ID
        metadata: Optional additional metadata

    Returns:
        True if logged successfully
    """
    tracker = get_tracker()

    if tracker is None:
        return False

    try:
        # Convert attention_map to regions format if provided
        attention_regions = None
        attention_entropy = 0.0
        if attention_map is not None:
            # If it's a dict with regions, use directly
            if isinstance(attention_map, dict) and 'regions' in attention_map:
                attention_regions = attention_map.get('regions')
                attention_entropy = attention_map.get('entropy', 0.0)

        record_id = tracker.log_prediction(
            image_id=prediction_id,  # Use prediction_id as image_id
            predicted_class=predicted_class,
            confidence=confidence,
            attention_regions=attention_regions,
            attention_entropy=attention_entropy,
            device_id=metadata.get('device_id', '') if metadata else '',
            session_id=user_id or ''
        )

        logger.debug(f"Logged prediction: {prediction_id} -> {predicted_class} ({confidence:.2%})")
        return True

    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
        return False


def log_correction(
    prediction_id: str,
    predicted_class: str,
    actual_class: str,
    is_correct: bool,
    confidence: float | None = None,
    attention_map: Any | None = None,
    metadata: dict | None = None
) -> bool:
    """
    Log a correction from user feedback.

    Call this when feedback is submitted.

    Args:
        prediction_id: ID of the original prediction (feedback_id)
        predicted_class: What the model predicted
        actual_class: What the user said it should be
        is_correct: True if prediction was correct
        confidence: Original prediction confidence
        attention_map: Optional attention map data
        metadata: Optional additional metadata

    Returns:
        True if logged successfully
    """
    tracker = get_tracker()

    if tracker is None:
        return False

    try:
        # Determine actual class - if correct, use predicted class
        final_actual_class = predicted_class if is_correct else actual_class

        record = tracker.log_correction(
            image_id=prediction_id,  # Use prediction_id as image_id
            actual_class=final_actual_class,
            corrector_id=metadata.get('user_id', '') if metadata else '',
            correction_source=metadata.get('source', 'feedback') if metadata else 'feedback'
        )

        if is_correct:
            logger.debug(f"Logged confirmation: {prediction_id} -> {predicted_class} ✓")
        else:
            logger.info(f"Logged correction: {prediction_id} -> {predicted_class} → {actual_class}")

        return record is not None

    except Exception as e:
        logger.error(f"Failed to log correction: {e}")
        return False


def get_summary() -> dict[str, Any]:
    """
    Get quick summary of correction statistics.

    Returns:
        Dictionary with summary statistics
    """
    tracker = get_tracker()

    if tracker is None:
        return {"error": "Analytics not initialized"}

    try:
        from .performance_analytics import PerformanceAnalytics
        analytics = PerformanceAnalytics(data_dir=str(tracker.data_dir))
        return analytics.get_overall_metrics()

    except Exception as e:
        logger.error(f"Failed to get summary: {e}")
        return {"error": str(e)}


def generate_report(output_dir: str | None = None) -> dict[str, Any]:
    """
    Generate comprehensive performance report.

    Args:
        output_dir: Directory to save report files

    Returns:
        Dictionary with paths to generated files
    """
    try:
        from .performance_dashboard import PerformanceDashboard
        dashboard = PerformanceDashboard()
        return dashboard.generate_full_report(output_dir=output_dir)

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return {"error": str(e)}


# Convenience exports
__all__ = [
    'init_analytics',
    'log_prediction',
    'log_correction',
    'get_tracker',
    'get_summary',
    'generate_report',
]
