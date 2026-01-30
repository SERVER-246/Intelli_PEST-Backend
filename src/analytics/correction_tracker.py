"""
CORRECTION TRACKER
==================
Core module for auto-logging predictions and corrections.

Storage: Dual JSON + SQLite for redundancy and query flexibility.
Logging: Continuous during server runtime.
Thread-safe: Uses locks for concurrent access.

Usage:
    from analytics import get_tracker, log_prediction, log_correction
    
    # Auto-log every prediction
    log_prediction(
        image_id="IMG_001.jpg",
        predicted_class="army worm",
        confidence=0.87,
        attention_regions=[(12, 0.15), (23, 0.12)]  # (region_id, score)
    )
    
    # Log when field worker provides correction
    log_correction(
        image_id="IMG_001.jpg",
        actual_class="Pink borer",  # Ground truth from field
        corrector_id="worker_42"
    )
"""

import json
import sqlite3
import threading
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PredictionStatus(str, Enum):
    """Status of a prediction."""
    PENDING = "pending"      # Awaiting field validation
    CORRECT = "correct"      # Field confirmed correct
    INCORRECT = "incorrect"  # Field provided correction
    FLAGGED = "flagged"      # Marked for review


@dataclass
class AttentionInfo:
    """Attention/region information for a prediction."""
    top_regions: List[Tuple[int, float]] = field(default_factory=list)  # (region_id, score)
    attention_entropy: float = 0.0  # How spread out attention is
    focus_area_pct: float = 0.0     # % of image where attention concentrated


@dataclass
class CorrectionRecord:
    """A single prediction/correction record."""
    # Identity
    record_id: str = ""
    image_id: str = ""
    image_hash: str = ""  # For deduplication
    
    # Prediction info
    predicted_class: str = ""
    predicted_confidence: float = 0.0
    model_version: str = ""
    prediction_timestamp: str = ""
    
    # Correction info (filled when correction arrives)
    actual_class: Optional[str] = None
    correction_timestamp: Optional[str] = None
    corrector_id: Optional[str] = None
    correction_source: str = ""  # "field", "expert", "review"
    
    # Status
    status: str = PredictionStatus.PENDING.value
    was_correct: Optional[bool] = None
    
    # Attention/region info
    attention_info: Optional[Dict] = None
    
    # Metadata
    location: str = ""
    device_id: str = ""
    session_id: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'CorrectionRecord':
        """Create from dictionary, filtering out unknown fields."""
        # Get the field names from the dataclass
        valid_fields = {f.name for f in fields(cls)}
        # Filter to only include valid fields
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


class CorrectionTracker:
    """
    Main tracker for predictions and corrections.
    
    Features:
    - Dual storage: JSON (human-readable backup) + SQLite (queryable)
    - Thread-safe operations
    - Auto-generates record IDs
    - Tracks model versions for comparison
    """
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize tracker.
        
        Args:
            data_dir: Directory for data storage. Defaults to feedback_data/analytics/
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "feedback_data" / "analytics"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.json_path = self.data_dir / "corrections_log.json"
        self.db_path = self.data_dir / "corrections.db"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory cache for quick access
        self._pending_predictions: Dict[str, CorrectionRecord] = {}
        
        # Current model version (set by server)
        self._model_version = "unknown"
        
        # Initialize storage
        self._init_database()
        self._load_pending_from_db()
        
        logger.info(f"CorrectionTracker initialized: {self.data_dir}")
    
    def set_model_version(self, version: str):
        """Set current model version for tracking."""
        self._model_version = version
        logger.info(f"Tracker model version set to: {version}")
    
    def _init_database(self):
        """Initialize SQLite database with schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Main corrections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS corrections (
                    record_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    image_hash TEXT,
                    predicted_class TEXT NOT NULL,
                    predicted_confidence REAL,
                    model_version TEXT,
                    prediction_timestamp TEXT,
                    actual_class TEXT,
                    correction_timestamp TEXT,
                    corrector_id TEXT,
                    correction_source TEXT,
                    status TEXT DEFAULT 'pending',
                    was_correct INTEGER,
                    attention_info TEXT,
                    location TEXT,
                    device_id TEXT,
                    session_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Indexes for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON corrections(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predicted_class ON corrections(predicted_class)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_actual_class ON corrections(actual_class)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_version ON corrections(model_version)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction_date ON corrections(prediction_timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_was_correct ON corrections(was_correct)')
            
            # Daily summary table (for quick reports)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date TEXT PRIMARY KEY,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    incorrect_predictions INTEGER DEFAULT 0,
                    pending_predictions INTEGER DEFAULT 0,
                    accuracy REAL,
                    model_version TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Per-class daily metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS class_daily_metrics (
                    date TEXT,
                    class_name TEXT,
                    predictions INTEGER DEFAULT 0,
                    correct INTEGER DEFAULT 0,
                    incorrect INTEGER DEFAULT 0,
                    avg_confidence REAL,
                    precision_score REAL,
                    recall_score REAL,
                    PRIMARY KEY (date, class_name)
                )
            ''')
            
            # Model version history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    version TEXT PRIMARY KEY,
                    first_seen TEXT,
                    last_seen TEXT,
                    total_predictions INTEGER DEFAULT 0,
                    overall_accuracy REAL
                )
            ''')
            
            conn.commit()
            conn.close()
    
    def _load_pending_from_db(self):
        """Load pending predictions into memory cache."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM corrections WHERE status = 'pending'
                ORDER BY prediction_timestamp DESC
                LIMIT 10000
            ''')
            
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                record_dict = dict(zip(columns, row))
                # Parse attention_info JSON
                if record_dict.get('attention_info'):
                    record_dict['attention_info'] = json.loads(record_dict['attention_info'])
                record = CorrectionRecord.from_dict(record_dict)
                self._pending_predictions[record.image_id] = record
            
            conn.close()
            logger.info(f"Loaded {len(self._pending_predictions)} pending predictions into cache")
    
    def _generate_record_id(self, image_id: str, timestamp: str) -> str:
        """Generate unique record ID."""
        data = f"{image_id}_{timestamp}_{self._model_version}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _compute_image_hash(self, image_data: Optional[bytes] = None) -> str:
        """Compute image hash for deduplication."""
        if image_data:
            return hashlib.md5(image_data).hexdigest()
        return ""
    
    def log_prediction(
        self,
        image_id: str,
        predicted_class: str,
        confidence: float,
        attention_regions: Optional[List[Tuple[int, float]]] = None,
        attention_entropy: float = 0.0,
        location: str = "",
        device_id: str = "",
        session_id: str = "",
        image_data: Optional[bytes] = None
    ) -> str:
        """
        Log a new prediction.
        
        Called automatically when inference is performed.
        
        Returns:
            record_id: Unique ID for this prediction record
        """
        with self._lock:
            timestamp = datetime.now().isoformat()
            record_id = self._generate_record_id(image_id, timestamp)
            
            # Build attention info
            attention_info = None
            if attention_regions:
                attention_info = {
                    "top_regions": attention_regions[:10],  # Top 10
                    "entropy": attention_entropy,
                    "focus_area_pct": sum(s for _, s in attention_regions[:5]) if attention_regions else 0
                }
            
            record = CorrectionRecord(
                record_id=record_id,
                image_id=image_id,
                image_hash=self._compute_image_hash(image_data),
                predicted_class=predicted_class,
                predicted_confidence=confidence,
                model_version=self._model_version,
                prediction_timestamp=timestamp,
                status=PredictionStatus.PENDING.value,
                attention_info=attention_info,
                location=location,
                device_id=device_id,
                session_id=session_id
            )
            
            # Save to database
            self._save_to_db(record)
            
            # Save to JSON backup
            self._save_to_json(record)
            
            # Cache in memory
            self._pending_predictions[image_id] = record
            
            # Update model version tracking
            self._update_model_version_stats()
            
            logger.debug(f"Logged prediction: {image_id} -> {predicted_class} ({confidence:.2%})")
            
            return record_id
    
    def log_correction(
        self,
        image_id: str,
        actual_class: str,
        corrector_id: str = "",
        correction_source: str = "field"
    ) -> Optional[CorrectionRecord]:
        """
        Log a correction for a previous prediction.
        
        Called when field worker provides ground truth.
        
        Returns:
            Updated CorrectionRecord, or None if image_id not found
        """
        with self._lock:
            # Find the pending prediction
            record = self._pending_predictions.get(image_id)
            
            if record is None:
                # Try to find in database
                record = self._find_latest_prediction(image_id)
            
            if record is None:
                logger.warning(f"No pending prediction found for: {image_id}")
                return None
            
            # Update record
            record.actual_class = actual_class
            record.correction_timestamp = datetime.now().isoformat()
            record.corrector_id = corrector_id
            record.correction_source = correction_source
            record.was_correct = (record.predicted_class.lower() == actual_class.lower())
            record.status = PredictionStatus.CORRECT.value if record.was_correct else PredictionStatus.INCORRECT.value
            
            # Update database
            self._update_in_db(record)
            
            # Update JSON
            self._update_in_json(record)
            
            # Remove from pending cache
            if image_id in self._pending_predictions:
                del self._pending_predictions[image_id]
            
            # Update daily summary
            self._update_daily_summary(record)
            
            # Update class metrics
            self._update_class_metrics(record)
            
            logger.info(
                f"Correction logged: {image_id} | "
                f"Predicted: {record.predicted_class} | "
                f"Actual: {actual_class} | "
                f"{'✓ CORRECT' if record.was_correct else '✗ INCORRECT'}"
            )
            
            return record
    
    def _save_to_db(self, record: CorrectionRecord):
        """Save record to SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        attention_json = json.dumps(record.attention_info) if record.attention_info else None
        
        cursor.execute('''
            INSERT OR REPLACE INTO corrections (
                record_id, image_id, image_hash, predicted_class, predicted_confidence,
                model_version, prediction_timestamp, actual_class, correction_timestamp,
                corrector_id, correction_source, status, was_correct, attention_info,
                location, device_id, session_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.record_id, record.image_id, record.image_hash,
            record.predicted_class, record.predicted_confidence,
            record.model_version, record.prediction_timestamp,
            record.actual_class, record.correction_timestamp,
            record.corrector_id, record.correction_source,
            record.status, 1 if record.was_correct else (0 if record.was_correct is False else None),
            attention_json, record.location, record.device_id, record.session_id,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_in_db(self, record: CorrectionRecord):
        """Update existing record in database."""
        self._save_to_db(record)  # INSERT OR REPLACE handles update
    
    def _save_to_json(self, record: CorrectionRecord):
        """Append record to JSON backup file."""
        try:
            # Load existing
            if self.json_path.exists():
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {"records": [], "last_updated": ""}
            
            # Add new record
            data["records"].append(record.to_dict())
            data["last_updated"] = datetime.now().isoformat()
            
            # Keep only last 50000 records in JSON (older are in SQLite)
            if len(data["records"]) > 50000:
                data["records"] = data["records"][-50000:]
            
            # Save
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
    
    def _update_in_json(self, record: CorrectionRecord):
        """Update record in JSON file."""
        try:
            if not self.json_path.exists():
                return
            
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            # Find and update record
            for i, r in enumerate(data["records"]):
                if r.get("record_id") == record.record_id:
                    data["records"][i] = record.to_dict()
                    break
            
            data["last_updated"] = datetime.now().isoformat()
            
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error updating JSON: {e}")
    
    def _find_latest_prediction(self, image_id: str) -> Optional[CorrectionRecord]:
        """Find the latest prediction for an image from database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM corrections 
            WHERE image_id = ? 
            ORDER BY prediction_timestamp DESC 
            LIMIT 1
        ''', (image_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            record_dict = dict(zip(columns, row))
            if record_dict.get('attention_info'):
                record_dict['attention_info'] = json.loads(record_dict['attention_info'])
            return CorrectionRecord.from_dict(record_dict)
        
        return None
    
    def _update_daily_summary(self, record: CorrectionRecord):
        """Update daily summary table."""
        date = record.prediction_timestamp[:10]  # YYYY-MM-DD
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get current counts
        cursor.execute('SELECT * FROM daily_summary WHERE date = ?', (date,))
        row = cursor.fetchone()
        
        if row:
            # Update existing
            cursor.execute('''
                UPDATE daily_summary SET
                    correct_predictions = correct_predictions + ?,
                    incorrect_predictions = incorrect_predictions + ?,
                    pending_predictions = pending_predictions - 1,
                    accuracy = CAST(correct_predictions + ? AS REAL) / 
                              NULLIF(correct_predictions + incorrect_predictions + ? + ?, 0),
                    updated_at = ?
                WHERE date = ?
            ''', (
                1 if record.was_correct else 0,
                0 if record.was_correct else 1,
                1 if record.was_correct else 0,
                1 if record.was_correct else 0,
                0 if record.was_correct else 1,
                datetime.now().isoformat(),
                date
            ))
        else:
            # Insert new
            cursor.execute('''
                INSERT INTO daily_summary (
                    date, total_predictions, correct_predictions, incorrect_predictions,
                    pending_predictions, accuracy, model_version, updated_at
                ) VALUES (?, 1, ?, ?, 0, ?, ?, ?)
            ''', (
                date,
                1 if record.was_correct else 0,
                0 if record.was_correct else 1,
                1.0 if record.was_correct else 0.0,
                record.model_version,
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def _update_class_metrics(self, record: CorrectionRecord):
        """Update per-class metrics."""
        date = record.prediction_timestamp[:10]
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Update predicted class metrics
        cursor.execute('''
            INSERT INTO class_daily_metrics (date, class_name, predictions, correct, incorrect, avg_confidence)
            VALUES (?, ?, 1, ?, ?, ?)
            ON CONFLICT(date, class_name) DO UPDATE SET
                predictions = predictions + 1,
                correct = correct + ?,
                incorrect = incorrect + ?,
                avg_confidence = (avg_confidence * predictions + ?) / (predictions + 1)
        ''', (
            date, record.predicted_class,
            1 if record.was_correct else 0,
            0 if record.was_correct else 1,
            record.predicted_confidence,
            1 if record.was_correct else 0,
            0 if record.was_correct else 1,
            record.predicted_confidence
        ))
        
        # If incorrect, also track the actual class
        if not record.was_correct and record.actual_class:
            cursor.execute('''
                INSERT INTO class_daily_metrics (date, class_name, predictions, correct, incorrect)
                VALUES (?, ?, 0, 0, 0)
                ON CONFLICT(date, class_name) DO UPDATE SET
                    predictions = predictions  -- No change, just ensure row exists
            ''', (date, record.actual_class))
        
        conn.commit()
        conn.close()
    
    def _update_model_version_stats(self):
        """Update model version tracking."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO model_versions (version, first_seen, last_seen, total_predictions)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(version) DO UPDATE SET
                last_seen = ?,
                total_predictions = total_predictions + 1
        ''', (self._model_version, now, now, now))
        
        conn.commit()
        conn.close()
    
    def get_pending_count(self) -> int:
        """Get count of pending predictions awaiting validation."""
        return len(self._pending_predictions)
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get quick stats summary."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Total counts
        cursor.execute('SELECT COUNT(*) FROM corrections')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM corrections WHERE was_correct = 1')
        correct = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM corrections WHERE was_correct = 0')
        incorrect = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM corrections WHERE status = "pending"')
        pending = cursor.fetchone()[0]
        
        conn.close()
        
        accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
        
        return {
            "total_predictions": total,
            "correct": correct,
            "incorrect": incorrect,
            "pending": pending,
            "accuracy": accuracy,
            "model_version": self._model_version
        }


# ============================================================
# Global Tracker Instance (Singleton)
# ============================================================

_tracker_instance: Optional[CorrectionTracker] = None
_tracker_lock = threading.Lock()


def get_tracker(data_dir: Optional[str] = None) -> CorrectionTracker:
    """Get or create the global tracker instance."""
    global _tracker_instance
    
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = CorrectionTracker(data_dir)
        return _tracker_instance


def log_prediction(
    image_id: str,
    predicted_class: str,
    confidence: float,
    attention_regions: Optional[List[Tuple[int, float]]] = None,
    **kwargs
) -> str:
    """Convenience function to log a prediction."""
    tracker = get_tracker()
    return tracker.log_prediction(
        image_id=image_id,
        predicted_class=predicted_class,
        confidence=confidence,
        attention_regions=attention_regions,
        **kwargs
    )


def log_correction(
    image_id: str,
    actual_class: str,
    corrector_id: str = "",
    correction_source: str = "field"
) -> Optional[CorrectionRecord]:
    """Convenience function to log a correction."""
    tracker = get_tracker()
    return tracker.log_correction(
        image_id=image_id,
        actual_class=actual_class,
        corrector_id=corrector_id,
        correction_source=correction_source
    )
