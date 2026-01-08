"""
Database Manager
================
Local SQLite database for user management and feedback tracking.
Replaces JSON file storage with proper relational database.
"""

import sqlite3
import logging
import threading
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class UserRecord:
    """User record from database."""
    user_id: str
    email: Optional[str] = None
    device_ids: str = "[]"  # JSON array
    total_submissions: int = 0
    total_feedbacks: int = 0
    correct_feedbacks: int = 0
    correction_feedbacks: int = 0
    first_seen: str = ""
    last_seen: str = ""
    locations: str = "[]"  # JSON array
    trust_score: float = 100.0
    is_flagged: bool = False
    flag_reason: Optional[str] = None
    flag_timestamp: Optional[str] = None
    corrections_by_class: str = "{}"  # JSON object
    user_type: str = "expert"  # expert, tester, admin
    notes: Optional[str] = None
    
    @property
    def correction_rate(self) -> float:
        if self.total_feedbacks == 0:
            return 0.0
        return self.correction_feedbacks / self.total_feedbacks
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["correction_rate"] = round(self.correction_rate, 4)
        data["device_ids"] = json.loads(self.device_ids)
        data["locations"] = json.loads(self.locations)
        data["corrections_by_class"] = json.loads(self.corrections_by_class)
        return data
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "UserRecord":
        return cls(
            user_id=row["user_id"],
            email=row["email"],
            device_ids=row["device_ids"],
            total_submissions=row["total_submissions"],
            total_feedbacks=row["total_feedbacks"],
            correct_feedbacks=row["correct_feedbacks"],
            correction_feedbacks=row["correction_feedbacks"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            locations=row["locations"],
            trust_score=row["trust_score"],
            is_flagged=bool(row["is_flagged"]),
            flag_reason=row["flag_reason"],
            flag_timestamp=row["flag_timestamp"],
            corrections_by_class=row["corrections_by_class"],
            user_type=row["user_type"],
            notes=row["notes"],
        )


class DatabaseManager:
    """
    SQLite database manager for user tracking and feedback storage.
    
    Tables:
    - users: User profiles and statistics
    - submissions: Individual submission records
    - feedback_entries: Feedback on predictions
    - image_metadata: Collected image information
    - audit_log: Changes and admin actions
    """
    
    def __init__(self, db_path: str = "./feedback_data/intellipest.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        
        # Initialize database
        self._init_database()
        
        logger.info(f"DatabaseManager initialized: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(str(self.db_path))
            self._local.connection.row_factory = sqlite3.Row
        
        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise e
    
    def _migrate_schema(self, cursor):
        """Migrate existing database schema to add missing columns."""
        # Get list of existing tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}
        
        # Define all table migrations: table_name -> [(column_name, column_type), ...]
        table_migrations = {
            'archived_images': [
                ('training_run_id', 'TEXT'),
                ('training_type', 'TEXT'),
                ('class_name', 'TEXT'),
                ('source_folder', 'TEXT'),
                ('archived_at', 'TEXT DEFAULT CURRENT_TIMESTAMP'),
            ],
            'users': [
                ('user_type', "TEXT DEFAULT 'expert'"),
            ],
            'system_events': [
                ('component', 'TEXT'),
                ('event_data', 'TEXT'),
                ('severity', "TEXT DEFAULT 'info'"),
                ('message', 'TEXT'),
            ],
            'scheduler_checks': [
                ('comprehensive_threshold_met', 'INTEGER DEFAULT 0'),
                ('training_type', 'TEXT'),
            ],
            'training_runs': [
                ('model_version_string', 'TEXT'),
                ('kd_enabled', 'INTEGER DEFAULT 0'),
                ('kd_teacher_count', 'INTEGER'),
                ('kd_teachers', 'TEXT'),
            ],
        }
        
        for table_name, columns in table_migrations.items():
            if table_name in existing_tables:
                cursor.execute(f"PRAGMA table_info({table_name})")
                existing_cols = {row[1] for row in cursor.fetchall()}
                
                for col_name, col_type in columns:
                    if col_name not in existing_cols:
                        try:
                            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
                            logger.info(f"Added column {col_name} to {table_name} table")
                        except sqlite3.OperationalError as e:
                            logger.warning(f"Could not add column {col_name} to {table_name}: {e}")
        
        logger.debug("Schema migration completed")
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT,
                    device_ids TEXT DEFAULT '[]',
                    total_submissions INTEGER DEFAULT 0,
                    total_feedbacks INTEGER DEFAULT 0,
                    correct_feedbacks INTEGER DEFAULT 0,
                    correction_feedbacks INTEGER DEFAULT 0,
                    first_seen TEXT,
                    last_seen TEXT,
                    locations TEXT DEFAULT '[]',
                    trust_score REAL DEFAULT 100.0,
                    is_flagged INTEGER DEFAULT 0,
                    flag_reason TEXT,
                    flag_timestamp TEXT,
                    corrections_by_class TEXT DEFAULT '{}',
                    user_type TEXT DEFAULT 'expert',
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Submissions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    image_hash TEXT NOT NULL,
                    predicted_class TEXT,
                    predicted_class_id INTEGER,
                    confidence REAL,
                    latitude REAL,
                    longitude REAL,
                    device_id TEXT,
                    app_version TEXT,
                    request_id TEXT,
                    submitted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Feedback entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE,
                    user_id TEXT,
                    image_hash TEXT,
                    predicted_class TEXT,
                    predicted_class_id INTEGER,
                    confidence REAL,
                    is_correct INTEGER,
                    corrected_class TEXT,
                    corrected_class_id INTEGER,
                    user_comment TEXT,
                    device_info TEXT,
                    app_version TEXT,
                    submitted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    processed INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Image metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_metadata (
                    image_hash TEXT PRIMARY KEY,
                    image_path TEXT,
                    predicted_class TEXT,
                    predicted_class_id INTEGER,
                    confidence REAL,
                    all_probabilities TEXT,
                    feedback_status TEXT DEFAULT 'unverified',
                    corrected_class TEXT,
                    corrected_class_id INTEGER,
                    user_id TEXT,
                    email TEXT,
                    device_id TEXT,
                    latitude REAL,
                    longitude REAL,
                    submission_timestamp TEXT,
                    feedback_timestamp TEXT,
                    user_trust_score REAL,
                    is_trusted_submission INTEGER DEFAULT 1,
                    request_id TEXT,
                    app_version TEXT,
                    original_filename TEXT,
                    file_size_bytes INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Audit log for tracking changes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    admin_id TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_submissions_user ON submissions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_submissions_image ON submissions(image_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback_entries(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_image ON feedback_entries(image_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_status ON image_metadata(feedback_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_type ON users(user_type)")
            
            # ==================== Training Tables ====================
            
            # Training runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    training_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    started_at TEXT,
                    completed_at TEXT,
                    model_version INTEGER,
                    epochs_planned INTEGER,
                    epochs_completed INTEGER DEFAULT 0,
                    total_images_used INTEGER DEFAULT 0,
                    images_per_class TEXT DEFAULT '{}',
                    junk_images_used INTEGER DEFAULT 0,
                    initial_upright_accuracy REAL,
                    final_upright_accuracy REAL,
                    best_upright_accuracy REAL,
                    initial_rotation_accuracy REAL,
                    final_rotation_accuracy REAL,
                    best_rotation_accuracy REAL,
                    early_stopped INTEGER DEFAULT 0,
                    collapse_detected INTEGER DEFAULT 0,
                    rollback_performed INTEGER DEFAULT 0,
                    backup_path TEXT,
                    error_message TEXT,
                    training_duration_seconds REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Training events/logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    epoch INTEGER,
                    batch INTEGER,
                    metric_name TEXT,
                    metric_value REAL,
                    message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                )
            """)
            
            # Archived images tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS archived_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_hash TEXT,
                    original_path TEXT,
                    archive_path TEXT,
                    training_run_id TEXT,
                    training_type TEXT,
                    class_name TEXT,
                    source_folder TEXT,
                    archived_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (training_run_id) REFERENCES training_runs(run_id)
                )
            """)
            
            # System events table (server start, scheduler, etc.)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    component TEXT,
                    event_data TEXT,
                    severity TEXT DEFAULT 'info',
                    message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Scheduler checks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scheduler_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    total_feedback_images INTEGER,
                    images_per_class TEXT,
                    classes_with_images INTEGER,
                    threshold_met INTEGER DEFAULT 0,
                    comprehensive_threshold_met INTEGER DEFAULT 0,
                    training_triggered INTEGER DEFAULT 0,
                    training_type TEXT,
                    reason TEXT
                )
            """)
            
            # Schema migration: Add missing columns to existing tables
            self._migrate_schema(cursor)
            
            # Create training indexes (only if columns exist)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_type ON training_runs(training_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_events_run ON training_events(run_id)")
            
            # Check if archived_images has training_run_id before creating index
            cursor.execute("PRAGMA table_info(archived_images)")
            archived_cols = [row[1] for row in cursor.fetchall()]
            if 'training_run_id' in archived_cols:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_archived_images_run ON archived_images(training_run_id)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type)")
            
            conn.commit()
            logger.info("Database schema initialized")
    
    # ==================== User Operations ====================
    
    def get_or_create_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        device_id: Optional[str] = None,
        user_type: str = "expert"
    ) -> UserRecord:
        """Get existing user or create new one."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            
            now = datetime.utcnow().isoformat() + "Z"
            
            if row is None:
                # Create new user
                device_ids = json.dumps([device_id] if device_id else [])
                cursor.execute("""
                    INSERT INTO users (user_id, email, device_ids, first_seen, last_seen, user_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, email, device_ids, now, now, user_type))
                conn.commit()
                
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                
                self._log_audit(conn, "user_created", "user", user_id)
            else:
                # Update existing user
                updates = [f"last_seen = '{now}'"]
                if email and not row["email"]:
                    updates.append(f"email = '{email}'")
                if device_id:
                    device_ids = json.loads(row["device_ids"])
                    if device_id not in device_ids:
                        device_ids.append(device_id)
                        updates.append(f"device_ids = '{json.dumps(device_ids)}'")
                
                cursor.execute(f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?", (user_id,))
                conn.commit()
                
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
            
            return UserRecord.from_row(row)
    
    def update_user_stats(
        self,
        user_id: str,
        submissions_delta: int = 0,
        feedbacks_delta: int = 0,
        correct_delta: int = 0,
        correction_delta: int = 0,
        trust_delta: float = 0.0,
        location: Optional[Dict[str, float]] = None,
        correction_class: Optional[str] = None,
    ):
        """Update user statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if not row:
                return
            
            # Build update
            updates = []
            now = datetime.utcnow().isoformat() + "Z"
            
            if submissions_delta:
                updates.append(f"total_submissions = total_submissions + {submissions_delta}")
            if feedbacks_delta:
                updates.append(f"total_feedbacks = total_feedbacks + {feedbacks_delta}")
            if correct_delta:
                updates.append(f"correct_feedbacks = correct_feedbacks + {correct_delta}")
            if correction_delta:
                updates.append(f"correction_feedbacks = correction_feedbacks + {correction_delta}")
            if trust_delta:
                new_trust = max(0, min(120, row["trust_score"] + trust_delta))
                updates.append(f"trust_score = {new_trust}")
            
            updates.append(f"last_seen = '{now}'")
            
            # Handle location update
            if location:
                locations = json.loads(row["locations"])
                locations.append({**location, "timestamp": now})
                if len(locations) > 100:
                    locations = locations[-100:]
                updates.append(f"locations = '{json.dumps(locations)}'")
            
            # Handle corrections by class
            if correction_class:
                corrections = json.loads(row["corrections_by_class"])
                corrections[correction_class] = corrections.get(correction_class, 0) + 1
                updates.append(f"corrections_by_class = '{json.dumps(corrections)}'")
            
            if updates:
                cursor.execute(f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?", (user_id,))
                conn.commit()
    
    def get_user(self, user_id: str) -> Optional[UserRecord]:
        """Get user by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            return UserRecord.from_row(row) if row else None
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users ORDER BY last_seen DESC")
            return [UserRecord.from_row(row).to_dict() for row in cursor.fetchall()]
    
    def get_flagged_users(self) -> List[Dict[str, Any]]:
        """Get all flagged users."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE is_flagged = 1")
            return [UserRecord.from_row(row).to_dict() for row in cursor.fetchall()]
    
    def unflag_user(self, user_id: str, admin_note: str = "") -> bool:
        """Unflag a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET 
                    is_flagged = 0,
                    flag_reason = ?,
                    trust_score = 100.0,
                    updated_at = ?
                WHERE user_id = ?
            """, (f"Unflagged: {admin_note}" if admin_note else None,
                  datetime.utcnow().isoformat() + "Z", user_id))
            
            if cursor.rowcount > 0:
                self._log_audit(conn, "user_unflagged", "user", user_id, notes=admin_note)
                conn.commit()
                return True
            return False
    
    def unflag_all_users(self, admin_note: str = "Disabled flagging - all users are trusted experts") -> int:
        """Unflag all users at once."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat() + "Z"
            
            cursor.execute("""
                UPDATE users SET 
                    is_flagged = 0,
                    flag_reason = ?,
                    trust_score = 100.0,
                    updated_at = ?
                WHERE is_flagged = 1
            """, (f"Unflagged: {admin_note}", now))
            
            count = cursor.rowcount
            if count > 0:
                self._log_audit(conn, "bulk_unflag", "users", None, 
                               new_value=str(count), notes=admin_note)
                conn.commit()
            
            return count
    
    def set_user_type(self, user_id: str, user_type: str) -> bool:
        """Set user type (expert, tester, admin)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET user_type = ?, updated_at = ?
                WHERE user_id = ?
            """, (user_type, datetime.utcnow().isoformat() + "Z", user_id))
            
            if cursor.rowcount > 0:
                self._log_audit(conn, "user_type_changed", "user", user_id, new_value=user_type)
                conn.commit()
                return True
            return False
    
    # ==================== Submission Operations ====================
    
    def record_submission(
        self,
        user_id: str,
        image_hash: str,
        predicted_class: Optional[str] = None,
        predicted_class_id: Optional[int] = None,
        confidence: Optional[float] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        device_id: Optional[str] = None,
        app_version: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> int:
        """Record a new submission."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO submissions (
                    user_id, image_hash, predicted_class, predicted_class_id,
                    confidence, latitude, longitude, device_id, app_version, request_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, image_hash, predicted_class, predicted_class_id,
                  confidence, latitude, longitude, device_id, app_version, request_id))
            conn.commit()
            return cursor.lastrowid
    
    # ==================== Feedback Operations ====================
    
    def record_feedback(
        self,
        feedback_id: str,
        user_id: str,
        image_hash: str,
        predicted_class: str,
        predicted_class_id: int,
        confidence: float,
        is_correct: bool,
        corrected_class: Optional[str] = None,
        corrected_class_id: Optional[int] = None,
        user_comment: Optional[str] = None,
        device_info: Optional[str] = None,
        app_version: Optional[str] = None,
    ) -> int:
        """Record feedback on a prediction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback_entries (
                    feedback_id, user_id, image_hash, predicted_class, predicted_class_id,
                    confidence, is_correct, corrected_class, corrected_class_id,
                    user_comment, device_info, app_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (feedback_id, user_id, image_hash, predicted_class, predicted_class_id,
                  confidence, 1 if is_correct else 0, corrected_class, corrected_class_id,
                  user_comment, device_info, app_version))
            conn.commit()
            return cursor.lastrowid
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM feedback_entries")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM feedback_entries WHERE is_correct = 1")
            correct = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM feedback_entries WHERE is_correct = 0")
            corrections = cursor.fetchone()[0]
            
            return {
                "total_feedback": total,
                "correct": correct,
                "corrections": corrections,
                "correction_rate": corrections / total if total > 0 else 0,
            }
    
    # ==================== Image Metadata Operations ====================
    
    def save_image_metadata(self, metadata: Dict[str, Any]):
        """Save or update image metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Convert dict fields to JSON strings
            all_probs = metadata.get("all_probabilities")
            if isinstance(all_probs, dict):
                all_probs = json.dumps(all_probs)
            
            cursor.execute("""
                INSERT OR REPLACE INTO image_metadata (
                    image_hash, image_path, predicted_class, predicted_class_id,
                    confidence, all_probabilities, feedback_status, corrected_class,
                    corrected_class_id, user_id, email, device_id, latitude, longitude,
                    submission_timestamp, feedback_timestamp, user_trust_score,
                    is_trusted_submission, request_id, app_version, original_filename,
                    file_size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.get("image_hash"),
                metadata.get("image_path"),
                metadata.get("predicted_class"),
                metadata.get("predicted_class_id"),
                metadata.get("confidence"),
                all_probs,
                metadata.get("feedback_status", "unverified"),
                metadata.get("corrected_class"),
                metadata.get("corrected_class_id"),
                metadata.get("user_id"),
                metadata.get("email"),
                metadata.get("device_id"),
                metadata.get("latitude"),
                metadata.get("longitude"),
                metadata.get("submission_timestamp"),
                metadata.get("feedback_timestamp"),
                metadata.get("user_trust_score"),
                1 if metadata.get("is_trusted_submission", True) else 0,
                metadata.get("request_id"),
                metadata.get("app_version"),
                metadata.get("original_filename"),
                metadata.get("file_size_bytes"),
            ))
            conn.commit()
    
    def get_image_metadata(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """Get image metadata by hash."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM image_metadata WHERE image_hash = ?", (image_hash,))
            row = cursor.fetchone()
            if row:
                data = dict(row)
                if data.get("all_probabilities"):
                    data["all_probabilities"] = json.loads(data["all_probabilities"])
                data["is_trusted_submission"] = bool(data["is_trusted_submission"])
                return data
            return None
    
    def get_training_images(
        self,
        include_correct: bool = True,
        include_corrected: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get images suitable for training."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            conditions = []
            if include_correct:
                conditions.append("feedback_status = 'correct'")
            if include_corrected:
                conditions.append("feedback_status = 'corrected'")
            
            if not conditions:
                return []
            
            query = f"SELECT * FROM image_metadata WHERE ({' OR '.join(conditions)})"
            cursor.execute(query)
            
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                if data.get("all_probabilities"):
                    data["all_probabilities"] = json.loads(data["all_probabilities"])
                results.append(data)
            
            return results
    
    # ==================== Training Operations ====================
    
    def create_training_run(
        self,
        run_id: str,
        training_type: str,
        model_version: int,
        epochs_planned: int,
        total_images: int = 0,
        images_per_class: Optional[Dict[str, int]] = None,
        junk_images: int = 0,
        model_version_string: Optional[str] = None,
        kd_enabled: bool = False,
        kd_teacher_count: Optional[int] = None,
        kd_teachers: Optional[List[str]] = None,
    ) -> int:
        """Create a new training run record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat() + "Z"
            cursor.execute("""
                INSERT INTO training_runs (
                    run_id, training_type, status, started_at, model_version,
                    epochs_planned, total_images_used, images_per_class, junk_images_used,
                    model_version_string, kd_enabled, kd_teacher_count, kd_teachers
                ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, training_type, now, model_version, epochs_planned,
                total_images, json.dumps(images_per_class or {}), junk_images,
                model_version_string, 1 if kd_enabled else 0, kd_teacher_count,
                json.dumps(kd_teachers) if kd_teachers else None
            ))
            conn.commit()
            logger.debug(f"Created training run {run_id} in database")
            return cursor.lastrowid
    
    def update_training_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        epochs_completed: Optional[int] = None,
        final_upright_accuracy: Optional[float] = None,
        final_rotation_accuracy: Optional[float] = None,
        best_upright_accuracy: Optional[float] = None,
        best_rotation_accuracy: Optional[float] = None,
        early_stopped: bool = False,
        collapse_detected: bool = False,
        rollback_performed: bool = False,
        error_message: Optional[str] = None,
        training_duration_seconds: Optional[float] = None,
        backup_path: Optional[str] = None,
        kd_teacher_count: Optional[int] = None,
        kd_teachers: Optional[List[str]] = None,
    ):
        """Update a training run record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if status:
                updates.append("status = ?")
                params.append(status)
                if status in ['completed', 'failed', 'cancelled']:
                    updates.append("completed_at = ?")
                    params.append(datetime.utcnow().isoformat() + "Z")
            
            if epochs_completed is not None:
                updates.append("epochs_completed = ?")
                params.append(epochs_completed)
            
            if final_upright_accuracy is not None:
                updates.append("final_upright_accuracy = ?")
                params.append(final_upright_accuracy)
            
            if final_rotation_accuracy is not None:
                updates.append("final_rotation_accuracy = ?")
                params.append(final_rotation_accuracy)
            
            if best_upright_accuracy is not None:
                updates.append("best_upright_accuracy = ?")
                params.append(best_upright_accuracy)
            
            if best_rotation_accuracy is not None:
                updates.append("best_rotation_accuracy = ?")
                params.append(best_rotation_accuracy)
            
            if early_stopped:
                updates.append("early_stopped = 1")
            
            if collapse_detected:
                updates.append("collapse_detected = 1")
            
            if rollback_performed:
                updates.append("rollback_performed = 1")
            
            if error_message:
                updates.append("error_message = ?")
                params.append(error_message)
            
            if training_duration_seconds is not None:
                updates.append("training_duration_seconds = ?")
                params.append(training_duration_seconds)
            
            if backup_path:
                updates.append("backup_path = ?")
                params.append(backup_path)
            
            if kd_teacher_count is not None:
                updates.append("kd_teacher_count = ?")
                params.append(kd_teacher_count)
            
            if kd_teachers is not None:
                updates.append("kd_teachers = ?")
                params.append(json.dumps(kd_teachers))
            
            if updates:
                params.append(run_id)
                cursor.execute(f"UPDATE training_runs SET {', '.join(updates)} WHERE run_id = ?", params)
                conn.commit()
    
    def log_training_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        event_data: Optional[Dict[str, Any]] = None,
    ):
        """Log a training event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_events (
                    run_id, event_type, message, epoch, batch,
                    metric_name, metric_value, event_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, event_type, message, epoch, batch,
                metric_name, metric_value, json.dumps(event_data) if event_data else None
            ))
            conn.commit()
    
    def log_archived_image(
        self,
        image_hash: str,
        original_path: str,
        archive_path: str,
        training_run_id: str,
        training_type: str,
        class_name: str,
        source_folder: str,
    ):
        """Log an archived image."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO archived_images (
                    image_hash, original_path, archive_path, training_run_id,
                    training_type, class_name, source_folder
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (image_hash, original_path, archive_path, training_run_id,
                  training_type, class_name, source_folder))
            conn.commit()
    
    def log_system_event(
        self,
        event_type: str,
        component: str,
        message: str,
        severity: str = "info",
        event_data: Optional[Dict[str, Any]] = None,
    ):
        """Log a system event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_events (event_type, component, message, severity, event_data)
                VALUES (?, ?, ?, ?, ?)
            """, (event_type, component, message, severity,
                  json.dumps(event_data) if event_data else None))
            conn.commit()
    
    def log_scheduler_check(
        self,
        total_feedback_images: int,
        images_per_class: Dict[str, int],
        classes_with_images: int,
        threshold_met: bool,
        comprehensive_threshold_met: bool,
        training_triggered: bool,
        training_type: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        """Log a scheduler check."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scheduler_checks (
                    total_feedback_images, images_per_class, classes_with_images,
                    threshold_met, comprehensive_threshold_met, training_triggered,
                    training_type, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                total_feedback_images, json.dumps(images_per_class), classes_with_images,
                1 if threshold_met else 0, 1 if comprehensive_threshold_met else 0,
                1 if training_triggered else 0, training_type, reason
            ))
            conn.commit()
    
    def get_training_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent training runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                if data.get("images_per_class"):
                    data["images_per_class"] = json.loads(data["images_per_class"])
                results.append(data)
            return results
    
    def get_training_events(self, run_id: str) -> List[Dict[str, Any]]:
        """Get events for a training run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM training_events WHERE run_id = ? ORDER BY created_at ASC
            """, (run_id,))
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                if data.get("event_data"):
                    data["event_data"] = json.loads(data["event_data"])
                results.append(data)
            return results
    
    def get_scheduler_checks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent scheduler checks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM scheduler_checks ORDER BY check_timestamp DESC LIMIT ?
            """, (limit,))
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                if data.get("images_per_class"):
                    data["images_per_class"] = json.loads(data["images_per_class"])
                results.append(data)
            return results
    
    def get_system_events(self, limit: int = 100, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get system events."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if severity:
                cursor.execute("""
                    SELECT * FROM system_events WHERE severity = ? ORDER BY created_at DESC LIMIT ?
                """, (severity, limit))
            else:
                cursor.execute("""
                    SELECT * FROM system_events ORDER BY created_at DESC LIMIT ?
                """, (limit,))
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                if data.get("event_data"):
                    data["event_data"] = json.loads(data["event_data"])
                results.append(data)
            return results
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total training runs by type
            cursor.execute("""
                SELECT training_type, status, COUNT(*) as count
                FROM training_runs GROUP BY training_type, status
            """)
            runs_by_type = {}
            for row in cursor.fetchall():
                type_key = row["training_type"]
                if type_key not in runs_by_type:
                    runs_by_type[type_key] = {}
                runs_by_type[type_key][row["status"]] = row["count"]
            
            # Average training duration
            cursor.execute("""
                SELECT training_type, AVG(training_duration_seconds) as avg_duration
                FROM training_runs WHERE status = 'completed'
                GROUP BY training_type
            """)
            avg_durations = {row["training_type"]: row["avg_duration"] for row in cursor.fetchall()}
            
            # Total images archived
            cursor.execute("SELECT COUNT(*) FROM archived_images")
            total_archived = cursor.fetchone()[0]
            
            # Total scheduler checks that triggered training
            cursor.execute("SELECT COUNT(*) FROM scheduler_checks WHERE training_triggered = 1")
            triggered_count = cursor.fetchone()[0]
            
            return {
                "runs_by_type": runs_by_type,
                "avg_durations": avg_durations,
                "total_archived_images": total_archived,
                "triggered_training_count": triggered_count,
            }
    
    # ==================== Audit Operations ====================
    
    def _log_audit(
        self,
        conn: sqlite3.Connection,
        action: str,
        entity_type: str,
        entity_id: Optional[str],
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        admin_id: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """Log an audit entry."""
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_log (action, entity_type, entity_id, old_value, new_value, admin_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (action, entity_type, entity_id, old_value, new_value, admin_id, notes))
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== Migration ====================
    
    def migrate_from_json(self, users_dir: str, metadata_dir: Optional[str] = None):
        """Migrate existing JSON data to database."""
        users_path = Path(users_dir)
        migrated_users = 0
        migrated_metadata = 0
        
        # Migrate users
        if users_path.exists():
            for filepath in users_path.glob("*.json"):
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    
                    with self._get_connection() as conn:
                        cursor = conn.cursor()
                        
                        # Check if user already exists
                        cursor.execute("SELECT user_id FROM users WHERE user_id = ?", 
                                      (data["user_id"],))
                        if cursor.fetchone():
                            continue
                        
                        cursor.execute("""
                            INSERT INTO users (
                                user_id, email, device_ids, total_submissions,
                                total_feedbacks, correct_feedbacks, correction_feedbacks,
                                first_seen, last_seen, locations, trust_score,
                                is_flagged, flag_reason, flag_timestamp, corrections_by_class,
                                user_type
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            data["user_id"],
                            data.get("email"),
                            json.dumps(data.get("device_ids", [])),
                            data.get("total_submissions", 0),
                            data.get("total_feedbacks", 0),
                            data.get("correct_feedbacks", 0),
                            data.get("correction_feedbacks", 0),
                            data.get("first_seen", ""),
                            data.get("last_seen", ""),
                            json.dumps(data.get("locations", [])),
                            100.0,  # Reset trust score (users are experts)
                            0,      # Not flagged (disabled)
                            None,   # No flag reason
                            None,   # No flag timestamp
                            json.dumps(data.get("corrections_by_class", {})),
                            "expert",  # All users are experts
                        ))
                        conn.commit()
                        migrated_users += 1
                except Exception as e:
                    logger.error(f"Failed to migrate user {filepath}: {e}")
        
        # Migrate metadata
        if metadata_dir:
            metadata_path = Path(metadata_dir)
            if metadata_path.exists():
                for filepath in metadata_path.glob("*.json"):
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)
                        
                        self.save_image_metadata(data)
                        migrated_metadata += 1
                    except Exception as e:
                        logger.error(f"Failed to migrate metadata {filepath}: {e}")
        
        self._log_audit(
            self._get_connection().__enter__(),
            "migration_complete",
            "system",
            None,
            notes=f"Migrated {migrated_users} users and {migrated_metadata} metadata records"
        )
        
        logger.info(f"Migration complete: {migrated_users} users, {migrated_metadata} metadata records")
        return {"users": migrated_users, "metadata": migrated_metadata}
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# Global instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> Optional[DatabaseManager]:
    """Get global database manager instance."""
    return _database_manager


def init_database_manager(db_path: str = "./feedback_data/intellipest.db") -> DatabaseManager:
    """Initialize global database manager."""
    global _database_manager
    _database_manager = DatabaseManager(db_path=db_path)
    return _database_manager
