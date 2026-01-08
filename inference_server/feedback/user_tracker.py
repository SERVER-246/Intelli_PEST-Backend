"""
User Tracker
============
Tracks user behavior and statistics.

NOTE: Automatic flagging has been DISABLED as all users are trusted experts
helping improve model generalization in the field. The flagging logic is kept
but disabled via FLAGGING_ENABLED = False for potential future use.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

# IMPORTANT: Flagging disabled - users are trusted experts
FLAGGING_ENABLED = False


@dataclass
class UserStats:
    """User statistics and behavior tracking."""
    user_id: str
    email: Optional[str] = None
    device_ids: List[str] = field(default_factory=list)
    
    # Submission stats
    total_submissions: int = 0
    total_feedbacks: int = 0
    correct_feedbacks: int = 0
    correction_feedbacks: int = 0
    
    # Timestamps
    first_seen: str = ""
    last_seen: str = ""
    
    # Location tracking
    locations: List[Dict[str, float]] = field(default_factory=list)
    
    # Trust system
    trust_score: float = 100.0
    is_flagged: bool = False
    flag_reason: Optional[str] = None
    flag_timestamp: Optional[str] = None
    
    # Suspicious patterns
    same_image_different_classes: int = 0
    rapid_submission_count: int = 0
    
    # Correction patterns
    corrections_by_class: Dict[str, int] = field(default_factory=dict)
    
    @property
    def correction_rate(self) -> float:
        """Calculate correction rate (how often user says model is wrong)."""
        if self.total_feedbacks == 0:
            return 0.0
        return self.correction_feedbacks / self.total_feedbacks
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["correction_rate"] = round(self.correction_rate, 4)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserStats":
        # Remove computed fields
        data.pop("correction_rate", None)
        return cls(**data)


@dataclass
class SubmissionRecord:
    """Record of a single submission for rate limiting."""
    timestamp: float
    image_hash: str
    location: Optional[Dict[str, float]] = None


class UserTracker:
    """
    Tracks users and their contributions.
    
    NOTE: Automatic flagging has been DISABLED.
    All users are treated as trusted experts helping improve model generalization.
    The flagging thresholds are kept for reference but not enforced.
    
    Original suspicious behaviors (NOW DISABLED):
    - High correction rate (>70%)
    - Same image with different "correct" classes
    - Rapid submissions (>10/min)
    - Always correcting to same class
    - Impossible location changes
    """
    
    # Thresholds (kept for reference, NOT ENFORCED when FLAGGING_ENABLED = False)
    CORRECTION_RATE_THRESHOLD = 0.70  # Flag if >70% corrections
    RAPID_SUBMISSION_THRESHOLD = 10   # Max submissions per minute
    SAME_IMAGE_DIFF_CLASS_THRESHOLD = 3  # Flag after 3 different classes for same image
    SINGLE_CLASS_CORRECTION_THRESHOLD = 0.80  # Flag if >80% corrections to one class
    MIN_FEEDBACKS_FOR_FLAG = 5  # Minimum feedbacks before flagging
    
    # Trust score adjustments
    CORRECT_FEEDBACK_BONUS = 1.0
    CORRECTION_PENALTY = 0.0  # DISABLED - corrections from experts are valuable
    RAPID_SUBMISSION_PENALTY = 0.0  # DISABLED - experts may submit rapidly
    DUPLICATE_DIFF_CLASS_PENALTY = 0.0  # DISABLED - experts may reconsider
    
    # Database integration
    _database = None
    
    def __init__(self, data_dir: str = "./feedback_data/users", use_database: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._users: Dict[str, UserStats] = {}
        self._recent_submissions: Dict[str, List[SubmissionRecord]] = defaultdict(list)
        self._image_corrections: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._lock = threading.Lock()
        
        # Initialize database if enabled
        self._use_database = use_database
        if use_database:
            try:
                from .database import init_database_manager, get_database_manager
                self._database = get_database_manager()
                if self._database is None:
                    db_path = self.data_dir.parent / "intellipest.db"
                    self._database = init_database_manager(str(db_path))
                logger.info("UserTracker using database backend")
            except Exception as e:
                logger.warning(f"Database init failed, falling back to JSON: {e}")
                self._use_database = False
        
        # Load existing data from JSON (for backwards compatibility)
        self._load_users()
        
        # Migrate to database if available
        if self._use_database and self._database:
            self._migrate_to_database()
        
        logger.info(f"UserTracker initialized. {len(self._users)} users loaded. Flagging: {'ENABLED' if FLAGGING_ENABLED else 'DISABLED'}")
    
    def get_or_create_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> UserStats:
        """Get existing user or create new one."""
        with self._lock:
            if user_id not in self._users:
                self._users[user_id] = UserStats(
                    user_id=user_id,
                    email=email,
                    device_ids=[device_id] if device_id else [],
                    first_seen=datetime.utcnow().isoformat() + "Z",
                    last_seen=datetime.utcnow().isoformat() + "Z",
                )
            else:
                user = self._users[user_id]
                user.last_seen = datetime.utcnow().isoformat() + "Z"
                if email and not user.email:
                    user.email = email
                if device_id and device_id not in user.device_ids:
                    user.device_ids.append(device_id)
            
            return self._users[user_id]
    
    def record_submission(
        self,
        user_id: str,
        image_hash: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        email: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record a new submission from user.
        
        Returns:
            Dict with trust_score and is_trusted flag
        """
        user = self.get_or_create_user(user_id, email, device_id)
        
        with self._lock:
            user.total_submissions += 1
            
            # Track location
            if latitude is not None and longitude is not None:
                user.locations.append({
                    "lat": latitude,
                    "lon": longitude,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
                # Keep only last 100 locations
                if len(user.locations) > 100:
                    user.locations = user.locations[-100:]
            
            # Check for rapid submissions
            now = time.time()
            recent = self._recent_submissions[user_id]
            recent.append(SubmissionRecord(
                timestamp=now,
                image_hash=image_hash,
                location={"lat": latitude, "lon": longitude} if latitude else None,
            ))
            
            # Clean old submissions (keep last 2 minutes)
            recent[:] = [r for r in recent if now - r.timestamp < 120]
            
            # Count submissions in last minute
            submissions_last_minute = sum(1 for r in recent if now - r.timestamp < 60)
            
            # NOTE: Rapid submission flagging DISABLED - users are trusted experts
            if FLAGGING_ENABLED and submissions_last_minute > self.RAPID_SUBMISSION_THRESHOLD:
                user.rapid_submission_count += 1
                user.trust_score = max(0, user.trust_score - self.RAPID_SUBMISSION_PENALTY)
                
                if user.rapid_submission_count >= 3 and not user.is_flagged:
                    self._flag_user(user, "Repeated rapid submissions detected")
            
            self._save_user(user)
        
        # All users are trusted (flagging disabled)
        return {
            "trust_score": user.trust_score,
            "is_trusted": True,  # Always trusted - users are experts
            "is_flagged": False,  # Never flagged - flagging disabled
        }
    
    def record_feedback(
        self,
        user_id: str,
        image_hash: str,
        is_correct: bool,
        predicted_class: str,
        correct_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record user feedback and check for suspicious patterns.
        
        Returns:
            Dict with trust status and any warnings
        """
        user = self.get_or_create_user(user_id)
        warnings = []
        
        with self._lock:
            user.total_feedbacks += 1
            
            if is_correct:
                user.correct_feedbacks += 1
                # Small trust bonus for correct feedback
                user.trust_score = min(120, user.trust_score + self.CORRECT_FEEDBACK_BONUS)
            else:
                user.correction_feedbacks += 1
                
                if correct_class:
                    # Track corrections by class
                    user.corrections_by_class[correct_class] = \
                        user.corrections_by_class.get(correct_class, 0) + 1
                    
                    # Track same image with different "correct" classes (for statistics only)
                    self._image_corrections[user_id][image_hash].add(correct_class)
                    
                    # NOTE: All flagging logic below is DISABLED - users are trusted experts
                    if FLAGGING_ENABLED and len(self._image_corrections[user_id][image_hash]) >= self.SAME_IMAGE_DIFF_CLASS_THRESHOLD:
                        user.same_image_different_classes += 1
                        user.trust_score = max(0, user.trust_score - self.DUPLICATE_DIFF_CLASS_PENALTY)
                        warnings.append("Same image corrected to multiple different classes")
                        
                        if not user.is_flagged:
                            self._flag_user(user, "Same image submitted with multiple different corrections")
                
                # NOTE: Penalty DISABLED - corrections from experts are valuable
                if FLAGGING_ENABLED and user.correction_rate > 0.5 and user.total_feedbacks >= self.MIN_FEEDBACKS_FOR_FLAG:
                    user.trust_score = max(0, user.trust_score - self.CORRECTION_PENALTY)
            
            # NOTE: High correction rate flagging DISABLED - experts may have high correction rates
            if FLAGGING_ENABLED and (user.total_feedbacks >= self.MIN_FEEDBACKS_FOR_FLAG and 
                user.correction_rate > self.CORRECTION_RATE_THRESHOLD and
                not user.is_flagged):
                self._flag_user(user, f"High correction rate: {user.correction_rate:.1%}")
                warnings.append(f"High correction rate detected: {user.correction_rate:.1%}")
            
            # NOTE: Single class correction flagging DISABLED - experts may specialize
            if FLAGGING_ENABLED and user.correction_feedbacks >= self.MIN_FEEDBACKS_FOR_FLAG:
                max_class_corrections = max(user.corrections_by_class.values()) if user.corrections_by_class else 0
                if max_class_corrections / user.correction_feedbacks > self.SINGLE_CLASS_CORRECTION_THRESHOLD:
                    if not user.is_flagged:
                        most_corrected = max(user.corrections_by_class, key=user.corrections_by_class.get)
                        self._flag_user(user, f"Always correcting to same class: {most_corrected}")
                        warnings.append(f"Pattern: Most corrections to '{most_corrected}'")
            
            self._save_user(user)
        
        # All users are trusted (flagging disabled)
        return {
            "trust_score": user.trust_score,
            "is_trusted": True,  # Always trusted - users are experts
            "is_flagged": False,  # Never flagged - flagging disabled
            "warnings": warnings,  # Still track patterns for analytics
        }
    
    def _flag_user(self, user: UserStats, reason: str):
        """Flag a user as suspicious. NOTE: Only called when FLAGGING_ENABLED=True"""
        if not FLAGGING_ENABLED:
            logger.debug(f"Flagging disabled - would have flagged {user.user_id}: {reason}")
            return
        user.is_flagged = True
        user.flag_reason = reason
        user.flag_timestamp = datetime.utcnow().isoformat() + "Z"
        user.trust_score = 0
        logger.warning(f"User flagged: {user.user_id} - Reason: {reason}")
    
    def unflag_user(self, user_id: str, admin_note: str = "") -> bool:
        """Admin action to unflag a user."""
        with self._lock:
            if user_id in self._users:
                user = self._users[user_id]
                user.is_flagged = False
                user.flag_reason = f"Unflagged by admin: {admin_note}" if admin_note else None
                user.trust_score = 100.0  # Reset to full trust (experts)
                self._save_user(user)
                logger.info(f"User unflagged by admin: {user_id}")
                return True
        return False
    
    def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific user."""
        with self._lock:
            if user_id in self._users:
                return self._users[user_id].to_dict()
        return None
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all user statistics."""
        with self._lock:
            return [u.to_dict() for u in self._users.values()]
    
    def get_flagged_users(self) -> List[Dict[str, Any]]:
        """Get all flagged users. NOTE: With flagging disabled, this returns empty list."""
        with self._lock:
            return [u.to_dict() for u in self._users.values() if u.is_flagged]
    
    def is_user_trusted(self, user_id: str) -> bool:
        """Check if user is trusted. NOTE: All users are trusted (flagging disabled)."""
        return True  # All users are trusted experts
    
    def unflag_all_users(self, admin_note: str = "Disabled flagging - all users are trusted experts") -> int:
        """Unflag all users at once. Called when disabling flagging system."""
        count = 0
        with self._lock:
            for user in self._users.values():
                if user.is_flagged:
                    user.is_flagged = False
                    user.flag_reason = f"Unflagged: {admin_note}"
                    user.trust_score = 100.0  # Reset to full trust
                    self._save_user(user)
                    count += 1
        logger.info(f"Unflagged {count} users: {admin_note}")
        return count
    
    def _migrate_to_database(self):
        """Migrate user data to database if available."""
        if not self._database:
            return
        
        try:
            migrated = self._database.migrate_from_json(
                str(self.data_dir),
                str(self.data_dir.parent / "metadata")
            )
            logger.info(f"Migrated to database: {migrated}")
        except Exception as e:
            logger.warning(f"Database migration failed: {e}")
    
    def _save_user(self, user: UserStats):
        """Save user data to file."""
        try:
            filepath = self.data_dir / f"{user.user_id}.json"
            with open(filepath, "w") as f:
                json.dump(user.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user {user.user_id}: {e}")
    
    def _load_users(self):
        """Load all users from files."""
        try:
            for filepath in self.data_dir.glob("*.json"):
                with open(filepath, "r") as f:
                    data = json.load(f)
                    user = UserStats.from_dict(data)
                    self._users[user.user_id] = user
        except Exception as e:
            logger.error(f"Error loading users: {e}")
    
    def export_to_excel(self, filepath: Optional[str] = None) -> str:
        """Export user data to Excel file."""
        try:
            import pandas as pd
        except ImportError:
            # Fallback to CSV if pandas not available
            return self.export_to_csv(filepath)
        
        filepath = filepath or str(self.data_dir / "user_records.xlsx")
        
        with self._lock:
            data = []
            for user in self._users.values():
                data.append({
                    "User ID": user.user_id,
                    "Email": user.email or "",
                    "Device IDs": ", ".join(user.device_ids),
                    "Total Submissions": user.total_submissions,
                    "Total Feedbacks": user.total_feedbacks,
                    "Correct": user.correct_feedbacks,
                    "Corrections": user.correction_feedbacks,
                    "Correction Rate": f"{user.correction_rate:.1%}",
                    "Trust Score": user.trust_score,
                    "Flagged": "YES" if user.is_flagged else "NO",
                    "Flag Reason": user.flag_reason or "",
                    "First Seen": user.first_seen,
                    "Last Seen": user.last_seen,
                })
            
            df = pd.DataFrame(data)
            df.to_excel(filepath, index=False, sheet_name="Users")
            
        logger.info(f"Exported user data to {filepath}")
        return filepath
    
    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """Export user data to CSV file."""
        import csv
        
        filepath = filepath or str(self.data_dir / "user_records.csv")
        
        with self._lock:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "User ID", "Email", "Device IDs", "Total Submissions",
                    "Total Feedbacks", "Correct", "Corrections", "Correction Rate",
                    "Trust Score", "Flagged", "Flag Reason", "First Seen", "Last Seen"
                ])
                
                for user in self._users.values():
                    writer.writerow([
                        user.user_id,
                        user.email or "",
                        ", ".join(user.device_ids),
                        user.total_submissions,
                        user.total_feedbacks,
                        user.correct_feedbacks,
                        user.correction_feedbacks,
                        f"{user.correction_rate:.1%}",
                        user.trust_score,
                        "YES" if user.is_flagged else "NO",
                        user.flag_reason or "",
                        user.first_seen,
                        user.last_seen,
                    ])
        
        logger.info(f"Exported user data to {filepath}")
        return filepath


# Global instance
_user_tracker: Optional[UserTracker] = None


def get_user_tracker() -> Optional[UserTracker]:
    """Get global user tracker instance."""
    return _user_tracker


def init_user_tracker(data_dir: str = "./feedback_data/users") -> UserTracker:
    """Initialize global user tracker."""
    global _user_tracker
    _user_tracker = UserTracker(data_dir=data_dir)
    return _user_tracker
