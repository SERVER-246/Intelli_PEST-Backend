"""
Model Version Manager
=====================
Semantic versioning for model releases.

Version Format: v{MAJOR}.{MINOR}.{PATCH}
- MAJOR: Manual architecture changes (reserved for future)
- MINOR: Comprehensive training (full 360° retraining)
- PATCH: Fine-tuning (incremental updates)

Examples:
- v1.0.0 → v1.0.1 (fine-tune)
- v1.0.1 → v1.0.2 (fine-tune)
- v1.0.2 → v1.1.0 (comprehensive - resets patch)
- v1.1.0 → v1.1.1 (fine-tune)
- v1.1.1 → v1.2.0 (comprehensive)
"""

import json
import logging
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a semantic version."""
    major: int = 1
    minor: int = 0
    patch: int = 0
    
    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"
    
    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)
    
    def to_dict(self) -> Dict[str, int]:
        return {"major": self.major, "minor": self.minor, "patch": self.patch}
    
    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        """Parse version from string like 'v1.2.3' or '1.2.3'."""
        clean = version_str.lstrip("v").strip()
        parts = clean.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2])
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "ModelVersion":
        return cls(
            major=data.get("major", 1),
            minor=data.get("minor", 0),
            patch=data.get("patch", 0)
        )
    
    def increment_patch(self) -> "ModelVersion":
        """Increment patch version (for fine-tuning)."""
        return ModelVersion(self.major, self.minor, self.patch + 1)
    
    def increment_minor(self) -> "ModelVersion":
        """Increment minor version, reset patch (for comprehensive training)."""
        return ModelVersion(self.major, self.minor + 1, 0)
    
    def increment_major(self) -> "ModelVersion":
        """Increment major version, reset minor and patch."""
        return ModelVersion(self.major + 1, 0, 0)


@dataclass
class VersionRecord:
    """Record of a model version."""
    version: str
    training_type: str  # "fine_tuning" or "comprehensive"
    created_at: str
    model_path: str
    backup_path: Optional[str] = None
    upright_accuracy: float = 0.0
    rotation_accuracy: float = 0.0
    images_used: int = 0
    training_duration_seconds: float = 0.0
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelVersionManager:
    """
    Manages semantic versioning for model files.
    
    Handles:
    - Version tracking and persistence
    - Model file naming (student_model_v1.2.3.pt)
    - Version history
    - Backup management
    """
    
    def __init__(
        self,
        model_dir: str = "D:/KnowledgeDistillation",
        backup_dir: str = "./model_backups",
        version_file: str = "./model_backups/version_info.json"
    ):
        self.model_dir = Path(model_dir)
        self.backup_dir = Path(backup_dir)
        self.version_file = Path(version_file)
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize version info
        self._version_info = self._load_version_info()
        
        logger.info(f"ModelVersionManager initialized - Current version: {self.current_version}")
    
    def _load_version_info(self) -> Dict[str, Any]:
        """Load version info from file or create default."""
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded version info: {data.get('current_version', 'v1.0.0')}")
                    return data
            except Exception as e:
                logger.warning(f"Could not load version info: {e}")
        
        # Default version info
        return {
            "current_version": "v1.0.0",
            "major": 1,
            "minor": 0,
            "patch": 0,
            "history": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    
    def _save_version_info(self):
        """Save version info to file."""
        self._version_info["updated_at"] = datetime.now().isoformat()
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self._version_info, f, indent=2)
            logger.debug(f"Version info saved: {self.current_version}")
        except Exception as e:
            logger.error(f"Could not save version info: {e}")
    
    @property
    def current_version(self) -> ModelVersion:
        """Get current model version."""
        return ModelVersion(
            major=self._version_info.get("major", 1),
            minor=self._version_info.get("minor", 0),
            patch=self._version_info.get("patch", 0)
        )
    
    @property
    def current_version_string(self) -> str:
        """Get current version as string."""
        return str(self.current_version)
    
    def get_model_filename(self, version: Optional[ModelVersion] = None) -> str:
        """Get model filename for a version."""
        v = version or self.current_version
        return f"student_model_{v}.pt"
    
    def get_model_path(self, version: Optional[ModelVersion] = None) -> Path:
        """Get full path to model file for a version."""
        return self.model_dir / self.get_model_filename(version)
    
    def get_current_model_path(self) -> Path:
        """Get path to current deployed model."""
        # Check for versioned model first
        versioned_path = self.get_model_path()
        if versioned_path.exists():
            return versioned_path
        
        # Fall back to legacy name
        legacy_path = self.model_dir / "student_model_rotation_robust.pt"
        if legacy_path.exists():
            return legacy_path
        
        # Last resort
        return self.model_dir / "student_model_final.pth"
    
    def increment_for_fine_tuning(self) -> ModelVersion:
        """
        Increment version for fine-tuning (PATCH +1).
        
        v1.0.0 → v1.0.1
        v1.2.3 → v1.2.4
        """
        new_version = self.current_version.increment_patch()
        self._update_version(new_version, "fine_tuning")
        logger.info(f"📦 Version incremented (fine-tuning): {self.current_version} → {new_version}")
        return new_version
    
    def increment_for_comprehensive(self) -> ModelVersion:
        """
        Increment version for comprehensive training (MINOR +1, PATCH = 0).
        
        v1.0.3 → v1.1.0
        v1.2.5 → v1.3.0
        """
        new_version = self.current_version.increment_minor()
        self._update_version(new_version, "comprehensive")
        logger.info(f"🚀 Version incremented (comprehensive): {self.current_version} → {new_version}")
        return new_version
    
    def increment_major(self) -> ModelVersion:
        """
        Increment major version (MAJOR +1, MINOR = 0, PATCH = 0).
        
        v1.5.3 → v2.0.0
        """
        new_version = self.current_version.increment_major()
        self._update_version(new_version, "major_release")
        logger.info(f"⭐ Version incremented (major): {self.current_version} → {new_version}")
        return new_version
    
    def _update_version(self, new_version: ModelVersion, training_type: str):
        """Update version info."""
        old_version = str(self.current_version)
        
        self._version_info["major"] = new_version.major
        self._version_info["minor"] = new_version.minor
        self._version_info["patch"] = new_version.patch
        self._version_info["current_version"] = str(new_version)
        
        # Add to history
        history_entry = {
            "from_version": old_version,
            "to_version": str(new_version),
            "training_type": training_type,
            "timestamp": datetime.now().isoformat(),
        }
        self._version_info.setdefault("history", []).append(history_entry)
        
        self._save_version_info()
    
    def record_training_complete(
        self,
        training_type: str,
        model_path: str,
        backup_path: Optional[str] = None,
        upright_accuracy: float = 0.0,
        rotation_accuracy: float = 0.0,
        images_used: int = 0,
        training_duration_seconds: float = 0.0,
        notes: Optional[str] = None,
    ):
        """Record that training completed for current version."""
        record = VersionRecord(
            version=str(self.current_version),
            training_type=training_type,
            created_at=datetime.now().isoformat(),
            model_path=model_path,
            backup_path=backup_path,
            upright_accuracy=upright_accuracy,
            rotation_accuracy=rotation_accuracy,
            images_used=images_used,
            training_duration_seconds=training_duration_seconds,
            notes=notes,
        )
        
        # Update last entry in history with full details
        if self._version_info.get("history"):
            self._version_info["history"][-1].update(record.to_dict())
        
        self._save_version_info()
        
        # Log to database if available
        self._log_to_database(record)
    
    def _log_to_database(self, record: VersionRecord):
        """Log version change to database."""
        try:
            from ..feedback.database import get_database_manager
            db = get_database_manager()
            if db:
                db.log_system_event(
                    event_type="version_change",
                    component="version_manager",
                    message=f"Model updated to {record.version} ({record.training_type})",
                    event_data=record.to_dict()
                )
        except Exception as e:
            logger.debug(f"Could not log version to database: {e}")
    
    def rename_model_with_version(
        self,
        source_path: Path,
        version: Optional[ModelVersion] = None,
        keep_original: bool = False
    ) -> Path:
        """
        Rename/copy model file to include version number.
        
        Args:
            source_path: Path to the trained model
            version: Version to use (defaults to current)
            keep_original: If True, copy instead of move
            
        Returns:
            Path to the versioned model file
        """
        v = version or self.current_version
        target_path = self.get_model_path(v)
        
        if not source_path.exists():
            logger.error(f"Source model not found: {source_path}")
            return source_path
        
        try:
            if keep_original:
                shutil.copy2(source_path, target_path)
                logger.info(f"Model copied: {source_path} → {target_path}")
            else:
                shutil.move(str(source_path), str(target_path))
                logger.info(f"Model renamed: {source_path} → {target_path}")
            
            return target_path
            
        except Exception as e:
            logger.error(f"Could not rename model: {e}")
            return source_path
    
    def get_version_history(self) -> list:
        """Get version history."""
        return self._version_info.get("history", [])
    
    def get_status(self) -> Dict[str, Any]:
        """Get version manager status."""
        return {
            "current_version": str(self.current_version),
            "major": self.current_version.major,
            "minor": self.current_version.minor,
            "patch": self.current_version.patch,
            "model_path": str(self.get_current_model_path()),
            "total_versions": len(self._version_info.get("history", [])),
            "last_updated": self._version_info.get("updated_at"),
        }


# Global instance
_version_manager: Optional[ModelVersionManager] = None


def get_version_manager() -> ModelVersionManager:
    """Get or create the global version manager."""
    global _version_manager
    if _version_manager is None:
        _version_manager = ModelVersionManager()
    return _version_manager


def init_version_manager(
    model_dir: str = "D:/KnowledgeDistillation",
    backup_dir: str = "./model_backups",
) -> ModelVersionManager:
    """Initialize the global version manager."""
    global _version_manager
    _version_manager = ModelVersionManager(
        model_dir=model_dir,
        backup_dir=backup_dir,
    )
    return _version_manager
