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
    CorrectionRecord,
    CorrectionTracker,
    get_tracker,
    log_correction,
    log_prediction,
)
from .integration import (
    get_summary,
    init_analytics,
)
from .integration import (
    log_correction as integration_log_correction,
)
from .integration import (
    log_prediction as integration_log_prediction,
)
from .performance_analytics import (
    ClassMetrics,
    ConfusionData,
    PerformanceAnalytics,
    TimeSeriesMetrics,
)
from .performance_dashboard import (
    PerformanceDashboard,
    export_csv,
    generate_charts,
    generate_report,
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
