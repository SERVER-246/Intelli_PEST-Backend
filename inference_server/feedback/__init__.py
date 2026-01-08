"""
Feedback Collection System
==========================
Collects user feedback on predictions for model improvement.

NOTE: User flagging has been DISABLED as all users are trusted experts
helping improve model generalization in the field.
"""

from .feedback_manager import FeedbackManager, FeedbackEntry, get_feedback_manager, init_feedback_manager
from .user_tracker import UserTracker, UserStats, get_user_tracker, init_user_tracker, FLAGGING_ENABLED
from .data_collector import DataCollector, ImageMetadata, get_data_collector, init_data_collector
from .database import DatabaseManager, get_database_manager, init_database_manager

__all__ = [
    "FeedbackManager", "FeedbackEntry", "get_feedback_manager", "init_feedback_manager",
    "UserTracker", "UserStats", "get_user_tracker", "init_user_tracker", "FLAGGING_ENABLED",
    "DataCollector", "ImageMetadata", "get_data_collector", "init_data_collector",
    "DatabaseManager", "get_database_manager", "init_database_manager",
]
