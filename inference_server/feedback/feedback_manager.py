"""
Feedback Manager
================
Manages collection and storage of user feedback on predictions.
"""

import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """Single feedback entry."""
    feedback_id: str
    prediction_id: str
    timestamp: str
    
    # Original prediction
    predicted_class: str
    predicted_class_id: int
    confidence: float
    
    # User feedback
    is_correct: Optional[bool] = None
    correct_class: Optional[str] = None
    correct_class_id: Optional[int] = None
    user_comment: Optional[str] = None
    
    # Image info (optional - for retraining)
    image_hash: Optional[str] = None
    image_path: Optional[str] = None
    
    # Metadata
    api_key_id: Optional[str] = None
    device_info: Optional[str] = None
    app_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackEntry":
        return cls(**data)


@dataclass
class PendingPrediction:
    """Prediction waiting for feedback."""
    prediction_id: str
    feedback_id: str
    timestamp: str
    predicted_class: str
    predicted_class_id: int
    confidence: float
    image_bytes: Optional[bytes] = None
    image_hash: Optional[str] = None
    user_id: Optional[str] = None
    expires_at: float = 0.0  # Unix timestamp


class FeedbackManager:
    """
    Manages feedback collection for model improvement.
    
    Features:
    - Generates feedback IDs for predictions
    - Stores pending predictions awaiting feedback
    - Saves feedback to JSON files
    - Optionally saves images for retraining
    - Auto-cleanup of expired pending predictions
    """
    
    def __init__(
        self,
        feedback_dir: str = "./feedback_data",
        images_dir: str = "./feedback_images",
        save_images: bool = True,
        pending_expiry_hours: int = 24,
        class_names: Optional[List[str]] = None,
    ):
        self.feedback_dir = Path(feedback_dir)
        self.images_dir = Path(images_dir)
        self.save_images = save_images
        self.pending_expiry_seconds = pending_expiry_hours * 3600
        self.class_names = class_names or []
        
        # Storage
        self._pending: Dict[str, PendingPrediction] = {}
        self._feedback: List[FeedbackEntry] = []
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "total_predictions": 0,
            "feedback_received": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "corrections_by_class": {},
            "junk_reports": 0,  # Count of images reported as junk/unrelated
            "special_categories": {},  # Count by special category type
        }
        
        # Class name aliases for flexible matching (display name -> canonical)
        # Maps common variations to the canonical class names used by the model
        self._class_aliases = {
            # Army worm variations
            "armyworm": "army worm",
            "army worm": "army worm",
            "fall armyworm": "army worm",
            # Porcupine
            "porcupine damage": "porcupine damage",
            "porcupine": "porcupine damage",
            # Rat
            "rat damage": "Rat damage",
            "rat": "Rat damage",
            # Mealy bug
            "mealy bug": "mealy bug",
            "mealybug": "mealy bug",
            # Borers
            "internode borer": "Internode borer",
            "pink borer": "Pink borer",
            "stalk borer": "Stalk borer",
            "top borer": "Top borer",
            "root borer": "root borer",
            # Others
            "termite": "termite",
            "healthy": "Healthy",
        }
        
        # Special feedback categories (not actual pest classes)
        # These are used for reporting images that shouldn't have been classified
        self._special_categories = {
            "junk": "JUNK",  # Unrelated image (not a plant/pest)
            "unrelated": "JUNK",  # Alias for junk
            "not applicable": "JUNK",  # Alias
            "n/a": "JUNK",  # Alias
            "other": "OTHER",  # Some other plant issue not in our classes
            "unknown": "UNKNOWN",  # Can't identify what it is
        }
        
        # Create directories
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        if self.save_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            # Create subdirectories for each class
            for class_name in self.class_names:
                (self.images_dir / "correct" / class_name).mkdir(parents=True, exist_ok=True)
                (self.images_dir / "corrected" / class_name).mkdir(parents=True, exist_ok=True)
        
        # Load existing feedback
        self._load_feedback()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"FeedbackManager initialized. Dir: {self.feedback_dir}")
    
    def set_class_names(self, class_names: List[str]):
        """Set class names for validation."""
        self.class_names = class_names
        # Create directories for new classes
        if self.save_images:
            for class_name in class_names:
                (self.images_dir / "correct" / class_name).mkdir(parents=True, exist_ok=True)
                (self.images_dir / "corrected" / class_name).mkdir(parents=True, exist_ok=True)
    
    def _match_class_name(self, input_class: str) -> Optional[str]:
        """
        Match user-provided class name to canonical class name.
        
        Handles case variations and common aliases.
        
        Args:
            input_class: Class name from user (e.g., "Armyworm", "army worm")
            
        Returns:
            Canonical class name if matched, None if invalid
        """
        if not input_class:
            return None
            
        # Normalize input: lowercase, strip whitespace
        normalized = input_class.lower().strip()
        
        # First, check exact match (case-insensitive)
        for class_name in self.class_names:
            if class_name.lower() == normalized:
                return class_name
        
        # Check aliases
        if normalized in self._class_aliases:
            alias_target = self._class_aliases[normalized]
            # Verify the alias target is in our class list
            for class_name in self.class_names:
                if class_name.lower() == alias_target.lower():
                    return class_name
        
        # Try partial match (user might submit "Armyworm" for "army worm")
        # Remove spaces and check
        no_space = normalized.replace(" ", "").replace("_", "")
        for class_name in self.class_names:
            if class_name.lower().replace(" ", "").replace("_", "") == no_space:
                return class_name
        
        # No match found
        return None
    
    def _match_special_category(self, input_class: str) -> Optional[str]:
        """
        Check if input is a special feedback category (not a pest class).
        
        Special categories include:
        - JUNK: Unrelated images that shouldn't have been classified
        - OTHER: Plant issues not in our pest classes
        - UNKNOWN: Cannot identify what the image shows
        
        Args:
            input_class: User input like "junk", "unrelated", "n/a"
            
        Returns:
            Category name (JUNK, OTHER, UNKNOWN) if matched, None otherwise
        """
        if not input_class:
            return None
        
        normalized = input_class.lower().strip()
        return self._special_categories.get(normalized)

    def register_prediction(
        self,
        predicted_class: str,
        predicted_class_id: int,
        confidence: float,
        image_bytes: Optional[bytes] = None,
        request_id: Optional[str] = None,
        image_hash: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Register a prediction for potential feedback.
        
        Returns:
            feedback_id: Unique ID for submitting feedback
        """
        prediction_id = request_id or str(uuid.uuid4())
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
        
        # Calculate image hash if not provided
        if not image_hash and image_bytes:
            import hashlib
            image_hash = hashlib.md5(image_bytes).hexdigest()
        
        pending = PendingPrediction(
            prediction_id=prediction_id,
            feedback_id=feedback_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            predicted_class=predicted_class,
            predicted_class_id=predicted_class_id,
            confidence=confidence,
            image_bytes=image_bytes if self.save_images else None,
            image_hash=image_hash,
            user_id=user_id,
            expires_at=time.time() + self.pending_expiry_seconds,
        )
        
        with self._lock:
            self._pending[feedback_id] = pending
            self._stats["total_predictions"] += 1
        
        logger.debug(f"Registered prediction {prediction_id} with feedback_id {feedback_id}")
        return feedback_id
    
    def submit_feedback(
        self,
        feedback_id: str,
        is_correct: bool,
        correct_class: Optional[str] = None,
        correct_class_id: Optional[int] = None,
        user_comment: Optional[str] = None,
        api_key_id: Optional[str] = None,
        device_info: Optional[str] = None,
        app_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit feedback for a prediction.
        
        Args:
            feedback_id: The feedback ID from prediction response
            is_correct: Whether the prediction was correct
            correct_class: The correct class name (if incorrect)
            correct_class_id: The correct class ID (if incorrect)
            user_comment: Optional user comment
            
        Returns:
            Dict with status and feedback entry
        """
        with self._lock:
            if feedback_id not in self._pending:
                return {
                    "status": "error",
                    "message": "Feedback ID not found or expired",
                    "code": "FEEDBACK_NOT_FOUND",
                }
            
            pending = self._pending.pop(feedback_id)
        
        # Validate correct_class if provided
        is_special_category = False
        special_category = None
        
        if not is_correct:
            if correct_class and self.class_names:
                # First check if it's a special category (junk, unrelated, etc.)
                special_category = self._match_special_category(correct_class)
                if special_category:
                    # This is a special category feedback
                    is_special_category = True
                    correct_class = special_category  # Use the category name
                    logger.info(f"Special category feedback: {special_category} for prediction {pending.predicted_class}")
                else:
                    # Try to match against pest classes
                    matched_class = self._match_class_name(correct_class)
                    if matched_class is None:
                        logger.warning(f"Invalid class name submitted: '{correct_class}', valid: {self.class_names}")
                        return {
                            "status": "error",
                            "message": f"Invalid class name: {correct_class}",
                            "code": "INVALID_CLASS",
                            "valid_classes": self.class_names + ["junk", "unrelated", "other", "unknown"],
                        }
                    correct_class = matched_class  # Use the canonical class name
            elif correct_class_id is not None and self.class_names:
                if 0 <= correct_class_id < len(self.class_names):
                    correct_class = self.class_names[correct_class_id]
                else:
                    return {
                        "status": "error",
                        "message": f"Invalid class ID: {correct_class_id}",
                        "code": "INVALID_CLASS_ID",
                        "valid_range": f"0-{len(self.class_names)-1}",
                    }
        
        # Track user feedback for suspicious behavior
        is_trusted = True
        from .user_tracker import get_user_tracker
        from .data_collector import get_data_collector
        
        user_tracker = get_user_tracker()
        if user_tracker and pending.user_id:
            track_result = user_tracker.record_feedback(
                user_id=pending.user_id,
                image_hash=pending.image_hash or "",
                is_correct=is_correct,
                predicted_class=pending.predicted_class,
                correct_class=correct_class,
            )
            is_trusted = track_result.get("is_trusted", True)
        
        # Update data collector with feedback
        data_collector = get_data_collector()
        if data_collector and pending.image_hash:
            data_collector.update_with_feedback(
                image_hash=pending.image_hash,
                is_correct=is_correct,
                corrected_class=correct_class,
                corrected_class_id=correct_class_id,
                is_trusted=is_trusted,
            )
        
        # Save image if available
        image_path = None
        if pending.image_bytes and self.save_images:
            # Determine if this is a junk report
            is_junk = is_special_category and special_category == "JUNK"
            
            image_path = self._save_feedback_image(
                pending.image_bytes,
                pending.image_hash,
                is_correct,
                correct_class if not is_correct else pending.predicted_class,
                is_junk=is_junk,
            )
        
        # Create feedback entry
        entry = FeedbackEntry(
            feedback_id=feedback_id,
            prediction_id=pending.prediction_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            predicted_class=pending.predicted_class,
            predicted_class_id=pending.predicted_class_id,
            confidence=pending.confidence,
            is_correct=is_correct,
            correct_class=correct_class if not is_correct else None,
            correct_class_id=correct_class_id if not is_correct else None,
            user_comment=user_comment,
            image_hash=pending.image_hash,
            image_path=image_path,
            api_key_id=api_key_id,
            device_info=device_info,
            app_version=app_version,
        )
        
        # Store and save
        with self._lock:
            self._feedback.append(entry)
            self._stats["feedback_received"] += 1
            
            if is_correct:
                self._stats["correct_predictions"] += 1
            else:
                self._stats["incorrect_predictions"] += 1
                # Track corrections by class or special category
                if correct_class:
                    if is_special_category:
                        # Track special category stats
                        self._stats["junk_reports"] += 1
                        self._stats["special_categories"][special_category] = \
                            self._stats["special_categories"].get(special_category, 0) + 1
                    else:
                        # Track class corrections
                        key = f"{pending.predicted_class} -> {correct_class}"
                        self._stats["corrections_by_class"][key] = \
                            self._stats["corrections_by_class"].get(key, 0) + 1
        
        self._save_feedback_entry(entry)
        
        logger.info(f"Feedback received: {feedback_id} - Correct: {is_correct}")
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "feedback_id": feedback_id,
            "recorded": {
                "is_correct": is_correct,
                "original_prediction": pending.predicted_class,
                "corrected_to": correct_class if not is_correct else None,
            },
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats["pending_feedbacks"] = len(self._pending)
            
            if stats["feedback_received"] > 0:
                stats["accuracy_from_feedback"] = round(
                    stats["correct_predictions"] / stats["feedback_received"] * 100, 2
                )
            else:
                stats["accuracy_from_feedback"] = None
            
            return stats
    
    def get_feedback_for_retraining(
        self,
        only_incorrect: bool = False,
        min_entries: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get feedback entries suitable for retraining.
        
        Returns entries with saved images that can be used for model improvement.
        """
        with self._lock:
            entries = []
            for entry in self._feedback:
                if entry.image_path and os.path.exists(entry.image_path):
                    if only_incorrect and entry.is_correct:
                        continue
                    entries.append(entry.to_dict())
            
            return entries if len(entries) >= min_entries else []
    
    def _save_feedback_image(
        self,
        image_bytes: bytes,
        image_hash: str,
        is_correct: bool,
        class_name: str,
        is_junk: bool = False,
    ) -> Optional[str]:
        """Save image to feedback directory for retraining."""
        try:
            # Determine save directory
            if is_junk:
                subdir = "junk"
                save_dir = self.images_dir / subdir
            else:
                subdir = "correct" if is_correct else "corrected"
                save_dir = self.images_dir / subdir / class_name
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Use hash + timestamp for unique filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{image_hash}_{timestamp}.jpg"
            filepath = save_dir / filename
            
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            
            # Notify retrain manager about new feedback image
            self._notify_retrain_manager(class_name, str(filepath), not is_correct, is_junk)
            
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save feedback image: {e}")
            return None
    
    def _notify_retrain_manager(
        self,
        class_name: str,
        image_path: str,
        is_correction: bool,
        is_junk: bool,
    ):
        """Notify the retrain manager about new feedback image."""
        try:
            from ..training.retrain_manager import get_retrain_manager
            
            retrain_manager = get_retrain_manager(
                model_path="D:/KnowledgeDistillation/student_model_rotation_robust.pt",
                feedback_dir=str(self.images_dir),
            )
            
            retrain_manager.record_feedback_image(
                class_name=class_name,
                image_path=image_path,
                is_correction=is_correction,
                is_junk=is_junk,
            )
        except Exception as e:
            logger.debug(f"Could not notify retrain manager: {e}")
    
    def _save_feedback_entry(self, entry: FeedbackEntry):
        """Save feedback entry to JSON file."""
        try:
            # Daily feedback file
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            filepath = self.feedback_dir / f"feedback_{date_str}.json"
            
            # Load existing or create new
            entries = []
            if filepath.exists():
                with open(filepath, "r") as f:
                    entries = json.load(f)
            
            entries.append(entry.to_dict())
            
            with open(filepath, "w") as f:
                json.dump(entries, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save feedback entry: {e}")
    
    def _load_feedback(self):
        """Load existing feedback from files."""
        try:
            for filepath in self.feedback_dir.glob("feedback_*.json"):
                with open(filepath, "r") as f:
                    entries = json.load(f)
                    for data in entries:
                        self._feedback.append(FeedbackEntry.from_dict(data))
                        
                        # Update stats
                        self._stats["feedback_received"] += 1
                        if data.get("is_correct"):
                            self._stats["correct_predictions"] += 1
                        else:
                            self._stats["incorrect_predictions"] += 1
            
            logger.info(f"Loaded {len(self._feedback)} existing feedback entries")
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
    
    def _start_cleanup_thread(self):
        """Start background thread to cleanup expired pending predictions."""
        def cleanup_loop():
            while True:
                time.sleep(3600)  # Check every hour
                self._cleanup_expired()
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    def _cleanup_expired(self):
        """Remove expired pending predictions."""
        now = time.time()
        with self._lock:
            expired = [
                fid for fid, p in self._pending.items()
                if p.expires_at < now
            ]
            for fid in expired:
                del self._pending[fid]
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired pending predictions")


# Global instance
_feedback_manager: Optional[FeedbackManager] = None


def get_feedback_manager() -> Optional[FeedbackManager]:
    """Get global feedback manager instance."""
    return _feedback_manager


def init_feedback_manager(
    feedback_dir: str = "./feedback_data",
    images_dir: str = "./feedback_images",
    save_images: bool = True,
    class_names: Optional[List[str]] = None,
) -> FeedbackManager:
    """Initialize global feedback manager."""
    global _feedback_manager
    _feedback_manager = FeedbackManager(
        feedback_dir=feedback_dir,
        images_dir=images_dir,
        save_images=save_images,
        class_names=class_names,
    )
    return _feedback_manager
