"""
Data Collector
==============
Silently collects images and metadata for model improvement.
"""

import hashlib
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Complete metadata for a collected image."""
    # Identifiers
    image_hash: str
    image_path: str
    
    # Prediction info
    predicted_class: str
    predicted_class_id: int
    confidence: float
    all_probabilities: Optional[Dict[str, float]] = None
    
    # User feedback
    feedback_status: str = "unverified"  # unverified, correct, corrected
    corrected_class: Optional[str] = None
    corrected_class_id: Optional[int] = None
    
    # User info
    user_id: Optional[str] = None
    email: Optional[str] = None
    device_id: Optional[str] = None
    
    # Location
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Timestamps
    submission_timestamp: str = ""
    feedback_timestamp: Optional[str] = None
    
    # Trust info
    user_trust_score: Optional[float] = None
    is_trusted_submission: bool = True
    
    # Request info
    request_id: Optional[str] = None
    app_version: Optional[str] = None
    
    # Image info
    original_filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageMetadata":
        return cls(**data)


class DataCollector:
    """
    Silently collects images and metadata for data enrichment.
    
    Folder structure:
    - images/correct/{class_name}/       - Confirmed correct predictions
    - images/corrected/{class_name}/     - User-corrected images (by correct class)
    - images/unverified/{date}/          - No feedback yet
    - images/flagged/{user_id}/          - From flagged users
    - metadata/{image_hash}.json         - Full metadata per image
    """
    
    def __init__(
        self,
        base_dir: str = "./feedback_data",
        class_names: Optional[List[str]] = None,
    ):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.metadata_dir = self.base_dir / "metadata"
        self.class_names = class_names or []
        
        self._lock = threading.Lock()
        self._metadata_cache: Dict[str, ImageMetadata] = {}
        
        # Create directory structure
        self._setup_directories()
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info(f"DataCollector initialized. Base dir: {self.base_dir}")
    
    def _setup_directories(self):
        """Create the directory structure."""
        # Main directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for image categories
        (self.images_dir / "correct").mkdir(exist_ok=True)
        (self.images_dir / "corrected").mkdir(exist_ok=True)
        (self.images_dir / "unverified").mkdir(exist_ok=True)
        (self.images_dir / "flagged").mkdir(exist_ok=True)
        
        # Create class subdirectories
        for class_name in self.class_names:
            (self.images_dir / "correct" / class_name).mkdir(exist_ok=True)
            (self.images_dir / "corrected" / class_name).mkdir(exist_ok=True)
    
    def set_class_names(self, class_names: List[str]):
        """Update class names and create directories."""
        self.class_names = class_names
        for class_name in class_names:
            (self.images_dir / "correct" / class_name).mkdir(exist_ok=True)
            (self.images_dir / "corrected" / class_name).mkdir(exist_ok=True)
    
    def collect_image(
        self,
        image_bytes: bytes,
        predicted_class: str,
        predicted_class_id: int,
        confidence: float,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        device_id: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        request_id: Optional[str] = None,
        app_version: Optional[str] = None,
        original_filename: Optional[str] = None,
        all_probabilities: Optional[Dict[str, float]] = None,
        user_trust_score: Optional[float] = None,
        is_flagged_user: bool = False,
    ) -> str:
        """
        Collect an image and its metadata silently.
        
        Returns:
            image_hash for later reference
        """
        # Calculate hash
        image_hash = hashlib.md5(image_bytes).hexdigest()
        
        # Determine save location
        if is_flagged_user:
            save_dir = self.images_dir / "flagged" / (user_id or "unknown")
        else:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            save_dir = self.images_dir / "unverified" / date_str
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        timestamp = datetime.utcnow().strftime("%H%M%S")
        filename = f"{image_hash}_{timestamp}.jpg"
        image_path = save_dir / filename
        
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        
        # Create metadata
        metadata = ImageMetadata(
            image_hash=image_hash,
            image_path=str(image_path),
            predicted_class=predicted_class,
            predicted_class_id=predicted_class_id,
            confidence=confidence,
            all_probabilities=all_probabilities,
            feedback_status="unverified",
            user_id=user_id,
            email=email,
            device_id=device_id,
            latitude=latitude,
            longitude=longitude,
            submission_timestamp=datetime.utcnow().isoformat() + "Z",
            user_trust_score=user_trust_score,
            is_trusted_submission=not is_flagged_user,
            request_id=request_id,
            app_version=app_version,
            original_filename=original_filename,
            file_size_bytes=len(image_bytes),
        )
        
        # Save metadata
        with self._lock:
            self._metadata_cache[image_hash] = metadata
            self._save_metadata(metadata)
        
        logger.debug(f"Collected image: {image_hash} from user {user_id}")
        return image_hash
    
    def update_with_feedback(
        self,
        image_hash: str,
        is_correct: bool,
        corrected_class: Optional[str] = None,
        corrected_class_id: Optional[int] = None,
        is_trusted: bool = True,
    ) -> bool:
        """
        Update image metadata and move to appropriate folder based on feedback.
        
        Returns:
            True if successful
        """
        with self._lock:
            if image_hash not in self._metadata_cache:
                logger.warning(f"Image hash not found: {image_hash}")
                return False
            
            metadata = self._metadata_cache[image_hash]
            old_path = Path(metadata.image_path)
            
            if not old_path.exists():
                logger.warning(f"Image file not found: {old_path}")
                return False
            
            # Update metadata
            metadata.feedback_timestamp = datetime.utcnow().isoformat() + "Z"
            metadata.is_trusted_submission = is_trusted
            
            if is_correct:
                metadata.feedback_status = "correct"
                # Move to correct/{predicted_class}/
                new_dir = self.images_dir / "correct" / metadata.predicted_class
            else:
                metadata.feedback_status = "corrected"
                metadata.corrected_class = corrected_class
                metadata.corrected_class_id = corrected_class_id
                # Move to corrected/{correct_class}/
                new_dir = self.images_dir / "corrected" / (corrected_class or "unknown")
            
            # Handle flagged/untrusted submissions
            if not is_trusted:
                new_dir = self.images_dir / "flagged" / (metadata.user_id or "unknown")
            
            new_dir.mkdir(parents=True, exist_ok=True)
            new_path = new_dir / old_path.name
            
            # Move file
            try:
                shutil.move(str(old_path), str(new_path))
                metadata.image_path = str(new_path)
            except Exception as e:
                logger.error(f"Failed to move image: {e}")
            
            # Save updated metadata
            self._save_metadata(metadata)
            
        return True
    
    def get_metadata(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an image."""
        with self._lock:
            if image_hash in self._metadata_cache:
                return self._metadata_cache[image_hash].to_dict()
        return None
    
    def get_training_data(
        self,
        include_correct: bool = True,
        include_corrected: bool = True,
        only_trusted: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get images suitable for retraining.
        
        Returns list of {image_path, class_name, confidence, is_corrected}
        """
        results = []
        
        with self._lock:
            for metadata in self._metadata_cache.values():
                if only_trusted and not metadata.is_trusted_submission:
                    continue
                
                if metadata.feedback_status == "correct" and include_correct:
                    results.append({
                        "image_path": metadata.image_path,
                        "class_name": metadata.predicted_class,
                        "class_id": metadata.predicted_class_id,
                        "confidence": metadata.confidence,
                        "is_corrected": False,
                        "user_verified": True,
                    })
                elif metadata.feedback_status == "corrected" and include_corrected:
                    results.append({
                        "image_path": metadata.image_path,
                        "class_name": metadata.corrected_class,
                        "class_id": metadata.corrected_class_id,
                        "confidence": metadata.confidence,
                        "is_corrected": True,
                        "user_verified": True,
                        "original_prediction": metadata.predicted_class,
                    })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        with self._lock:
            stats = {
                "total_collected": len(self._metadata_cache),
                "unverified": 0,
                "correct": 0,
                "corrected": 0,
                "trusted": 0,
                "untrusted": 0,
                "by_class": {},
            }
            
            for metadata in self._metadata_cache.values():
                stats[metadata.feedback_status] = stats.get(metadata.feedback_status, 0) + 1
                
                if metadata.is_trusted_submission:
                    stats["trusted"] += 1
                else:
                    stats["untrusted"] += 1
                
                # Count by class
                class_name = metadata.corrected_class or metadata.predicted_class
                if class_name not in stats["by_class"]:
                    stats["by_class"][class_name] = {"total": 0, "correct": 0, "corrected": 0}
                stats["by_class"][class_name]["total"] += 1
                if metadata.feedback_status == "correct":
                    stats["by_class"][class_name]["correct"] += 1
                elif metadata.feedback_status == "corrected":
                    stats["by_class"][class_name]["corrected"] += 1
            
            return stats
    
    def _save_metadata(self, metadata: ImageMetadata):
        """Save metadata to JSON file."""
        try:
            filepath = self.metadata_dir / f"{metadata.image_hash}.json"
            with open(filepath, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _load_metadata(self):
        """Load all metadata from files."""
        try:
            for filepath in self.metadata_dir.glob("*.json"):
                with open(filepath, "r") as f:
                    data = json.load(f)
                    metadata = ImageMetadata.from_dict(data)
                    self._metadata_cache[metadata.image_hash] = metadata
            logger.info(f"Loaded {len(self._metadata_cache)} image metadata records")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
    
    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """Export metadata to CSV for analysis."""
        import csv
        
        filepath = filepath or str(self.base_dir / "image_metadata.csv")
        
        with self._lock:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Image Hash", "Predicted Class", "Confidence", "Feedback Status",
                    "Corrected Class", "User ID", "Email", "Latitude", "Longitude",
                    "Submission Time", "Feedback Time", "Trust Score", "Trusted",
                    "Image Path"
                ])
                
                for m in self._metadata_cache.values():
                    writer.writerow([
                        m.image_hash,
                        m.predicted_class,
                        f"{m.confidence:.4f}",
                        m.feedback_status,
                        m.corrected_class or "",
                        m.user_id or "",
                        m.email or "",
                        m.latitude or "",
                        m.longitude or "",
                        m.submission_timestamp,
                        m.feedback_timestamp or "",
                        m.user_trust_score or "",
                        "Yes" if m.is_trusted_submission else "No",
                        m.image_path,
                    ])
        
        logger.info(f"Exported metadata to {filepath}")
        return filepath


# Global instance
_data_collector: Optional[DataCollector] = None


def get_data_collector() -> Optional[DataCollector]:
    """Get global data collector instance."""
    return _data_collector


def init_data_collector(
    base_dir: str = "./feedback_data",
    class_names: Optional[List[str]] = None,
) -> DataCollector:
    """Initialize global data collector."""
    global _data_collector
    _data_collector = DataCollector(base_dir=base_dir, class_names=class_names)
    return _data_collector
