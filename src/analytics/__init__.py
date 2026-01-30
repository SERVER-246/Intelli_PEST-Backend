"""
Intelli-PEST Analytics Module
==============================
Performance correction tracking, analytics, and dashboard system.

Components:
- CorrectionTracker: Auto-logs corrections during server runtime
- PerformanceAnalytics: Computes accuracy, precision, trends
- PerformanceDashboard: Generates charts, reports, exports
- Integration: Easy integration into inference server
"""

from .correction_tracker import (
    CorrectionTracker,
    CorrectionRecord,
    get_tracker,
    log_prediction,
    log_correction,
)

from .performance_analytics import (
    PerformanceAnalytics,
    ClassMetrics,
    TimeSeriesMetrics,
    ConfusionData,
)

from .performance_dashboard import (
    PerformanceDashboard,
    generate_report,
    generate_charts,
    export_csv,
)

from .integration import (
    init_analytics,
    log_prediction as integration_log_prediction,
    log_correction as integration_log_correction,
    get_summary,
)

__all__ = [
    # Tracker
    "CorrectionTracker",
    "CorrectionRecord", 
    "get_tracker",
    "log_prediction",
    "log_correction",
    # Analytics
    "PerformanceAnalytics",
    "ClassMetrics",
    "TimeSeriesMetrics",
    "ConfusionData",
    # Dashboard
    "PerformanceDashboard",
    "generate_report",
    "generate_charts",
    "export_csv",
    # Integration
    "init_analytics",
    "get_summary",
]
