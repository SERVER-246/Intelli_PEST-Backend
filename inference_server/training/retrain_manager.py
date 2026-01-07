"""
Model Retraining Manager
========================
Manages automatic model retraining based on user feedback.

Features:
- Tracks feedback image counts per class
- Triggers retraining when thresholds are met
- Backs up current model before retraining
- Fine-tunes model with feedback images
- Updates deployment model after successful training
"""

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import hashlib

logger = logging.getLogger(__name__)

# Lazy imports for training
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    pass

PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    pass


@dataclass
class RetrainingConfig:
    """Configuration for model retraining."""
    # Thresholds - BALANCED requirements to prevent bias
    min_images_per_class: int = 10  # Trigger when any class reaches this
    min_total_images: int = 150  # Trigger when total reaches this
    min_classes_with_images: int = 3  # Require at least N classes to have images (prevents bias)
    max_class_imbalance_ratio: float = 5.0  # Max ratio between largest and smallest class
    
    # Training parameters (matching rotational training script)
    epochs: int = 15  # Fine-tuning epochs
    batch_size: int = 16
    learning_rate: float = 5e-4  # 0.0005 - same as rotation finetuning
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01  # Matching original training
    temperature: float = 4.0  # For soft label consistency
    
    # Elastic Weight Consolidation (prevents catastrophic forgetting)
    use_ewc: bool = True
    ewc_lambda: float = 100.0  # Reduced from 500 to prevent explosion (matching rotation script)
    max_ewc_loss: float = 1.0  # Clamp EWC loss to prevent dominating training
    
    # Gradient clipping (CRITICAL - prevents exploding gradients)
    gradient_clip_norm: float = 1.0
    
    # Learning rate schedule
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2  # Warmup for first N epochs (matching rotation script)
    
    # Early stopping (dual-metric like rotation training)
    patience: int = 5  # Stop if no improvement for N epochs
    min_accuracy_improvement: float = 0.1  # Minimum improvement to count as progress
    
    # Collapse detection
    collapse_threshold: float = 30.0  # Accuracy drop > this triggers rollback
    
    # Rotation robustness
    rotation_prob: float = 0.8  # Probability of rotation augmentation (matching training)
    validate_rotation: bool = True  # Track both upright and rotated accuracy
    
    # Class balancing
    use_class_weights: bool = True  # Weight loss by inverse class frequency
    oversample_minority: bool = True  # Oversample smaller classes
    
    # Paths
    model_path: str = ""  # Current model path
    backup_dir: str = "./model_backups"
    feedback_images_dir: str = "./feedback_data/images"
    training_history_dir: str = "./model_backups/history"  # Archive of used training images
    
    # Options
    include_junk_class: bool = True  # Add "junk" as 12th class
    freeze_backbone_epochs: int = 3  # Freeze backbone for first N epochs, then unfreeze
    
    # Mixed precision (matching training)
    use_amp: bool = True
    

@dataclass
class RetrainingHistory:
    """Record of a single retraining event."""
    version: int
    timestamp: str
    images_used: int
    images_per_class: Dict[str, int]
    junk_images_used: int
    epochs: int
    backup_path: str
    success: bool
    error: Optional[str] = None
    training_duration_seconds: float = 0.0
    # Dual metrics (matching rotation training)
    final_upright_accuracy: float = 0.0
    final_rotation_accuracy: float = 0.0
    best_upright_accuracy: float = 0.0
    best_rotation_accuracy: float = 0.0
    collapse_detected: bool = False
    early_stopped: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrainingStatus:
    """Status of retraining process."""
    is_training: bool = False
    last_trained: Optional[str] = None
    last_backup_path: Optional[str] = None
    training_progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    error: Optional[str] = None
    
    # Version tracking
    current_version: int = 1  # Model version (increments with each retrain)
    total_retrains: int = 0  # Total number of retraining runs
    
    # Image counts (NEW images since last retrain)
    total_feedback_images: int = 0
    images_per_class: Dict[str, int] = field(default_factory=dict)
    junk_images: int = 0
    
    # Thresholds
    threshold_per_class: int = 10
    threshold_total: int = 150
    threshold_min_classes: int = 3  # Minimum classes with images to prevent bias
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def classes_with_images(self) -> int:
        """Count how many classes have at least 1 image."""
        return sum(1 for count in self.images_per_class.values() if count > 0)
    
    @property
    def ready_to_retrain(self) -> bool:
        """
        Check if thresholds are met for retraining.
        
        Requirements:
        1. Either per-class threshold OR total threshold met
        2. At least min_classes have images (prevents bias from single-class data)
        """
        # First check class diversity (prevents bias)
        if self.classes_with_images < self.threshold_min_classes:
            return False
        
        # Check if any class has enough images
        for count in self.images_per_class.values():
            if count >= self.threshold_per_class:
                return True
        # Check total
        return self.total_feedback_images >= self.threshold_total
    
    @property
    def retrain_blocked_reason(self) -> Optional[str]:
        """Return reason why retraining is blocked, or None if ready."""
        if self.classes_with_images < self.threshold_min_classes:
            return f"Need images from at least {self.threshold_min_classes} classes (have {self.classes_with_images}) to prevent model bias"
        
        has_class_threshold = any(count >= self.threshold_per_class for count in self.images_per_class.values())
        has_total_threshold = self.total_feedback_images >= self.threshold_total
        
        if not has_class_threshold and not has_total_threshold:
            return f"Need {self.threshold_per_class} images in one class OR {self.threshold_total} total images"
        
        return None


class FeedbackImageDataset(Dataset):
    """Dataset for loading feedback images."""
    
    def __init__(
        self,
        images_dir: Path,
        class_names: List[str],
        include_junk: bool = True,
        transform=None
    ):
        self.images_dir = Path(images_dir)
        self.class_names = list(class_names)
        self.include_junk = include_junk
        
        if include_junk and "junk" not in [c.lower() for c in self.class_names]:
            self.class_names.append("junk")
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and labels."""
        self.samples = []
        
        # Load from 'corrected' folder (user-corrected predictions)
        corrected_dir = self.images_dir / "corrected"
        if corrected_dir.exists():
            for class_dir in corrected_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    if class_name in self.class_names:
                        class_idx = self.class_names.index(class_name)
                        for img_path in class_dir.glob("*"):
                            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                self.samples.append((str(img_path), class_idx))
        
        # Load from 'correct' folder (confirmed correct predictions)
        correct_dir = self.images_dir / "correct"
        if correct_dir.exists():
            for class_dir in correct_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    if class_name in self.class_names:
                        class_idx = self.class_names.index(class_name)
                        for img_path in class_dir.glob("*"):
                            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                self.samples.append((str(img_path), class_idx))
        
        # Load junk images
        if self.include_junk:
            junk_dir = self.images_dir / "junk"
            if junk_dir.exists():
                junk_idx = self.class_names.index("junk")
                for img_path in junk_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((str(img_path), junk_idx))
        
        logger.info(f"Loaded {len(self.samples)} feedback images for retraining")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return a placeholder
            return torch.zeros(3, 256, 256), label


class ModelRetrainingManager:
    """
    Manages automatic model retraining based on feedback.
    
    Workflow:
    1. Tracks feedback image counts per class
    2. When thresholds are met, triggers retraining
    3. Backs up current model
    4. Fine-tunes model with feedback images
    5. Updates deployment model
    """
    
    CLASS_NAMES = [
        "Healthy",
        "Internode borer",
        "Pink borer",
        "Rat damage",
        "Stalk borer",
        "Top borer",
        "army worm",
        "mealy bug",
        "porcupine damage",
        "root borer",
        "termite",
    ]
    
    def __init__(
        self,
        config: Optional[RetrainingConfig] = None,
        model_path: Optional[str] = None,
        feedback_dir: Optional[str] = None,
    ):
        """
        Initialize retraining manager.
        
        Args:
            config: Retraining configuration
            model_path: Path to current deployed model
            feedback_dir: Path to feedback images directory
        """
        self.config = config or RetrainingConfig()
        
        if model_path:
            self.config.model_path = model_path
        if feedback_dir:
            self.config.feedback_images_dir = feedback_dir
        
        self._status = RetrainingStatus(
            threshold_per_class=self.config.min_images_per_class,
            threshold_total=self.config.min_total_images,
            threshold_min_classes=self.config.min_classes_with_images,
        )
        
        # EWC Fisher information (for preventing catastrophic forgetting)
        self._fisher_information: Optional[Dict[str, torch.Tensor]] = None
        self._optimal_params: Optional[Dict[str, torch.Tensor]] = None
        
        self._lock = threading.Lock()
        self._training_thread: Optional[threading.Thread] = None
        self._on_retrain_complete: Optional[Callable] = None
        
        # Status file for persistence across restarts
        self._status_file = Path(self.config.backup_dir) / "retrain_status.json"
        
        # Create directories
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.training_history_dir).mkdir(parents=True, exist_ok=True)
        
        # Load persisted status (last_trained, etc.)
        self._load_status()
        
        # Update image counts from disk
        self._update_image_counts()
        
        logger.info(f"ModelRetrainingManager initialized")
        logger.info(f"  Model path: {self.config.model_path}")
        logger.info(f"  Feedback dir: {self.config.feedback_images_dir}")
        logger.info(f"  Version: {self._status.current_version}, Total retrains: {self._status.total_retrains}")
        logger.info(f"  Thresholds: {self.config.min_images_per_class}/class or {self.config.min_total_images} total, min {self.config.min_classes_with_images} classes")
    
    def _load_status(self):
        """Load persisted status from disk (survives server restarts)."""
        try:
            if self._status_file.exists():
                with open(self._status_file, 'r') as f:
                    data = json.load(f)
                    self._status.last_trained = data.get('last_trained')
                    self._status.last_backup_path = data.get('last_backup_path')
                    self._status.current_version = data.get('current_version', 1)
                    self._status.total_retrains = data.get('total_retrains', 0)
                    logger.info(f"Loaded persisted status: version={self._status.current_version}, "
                               f"retrains={self._status.total_retrains}, last_trained={self._status.last_trained}")
        except Exception as e:
            logger.warning(f"Could not load status file: {e}")
    
    def _save_status(self):
        """Save status to disk for persistence across restarts."""
        try:
            data = {
                'last_trained': self._status.last_trained,
                'last_backup_path': self._status.last_backup_path,
                'current_version': self._status.current_version,
                'total_retrains': self._status.total_retrains,
                'updated_at': datetime.now().isoformat(),
            }
            with open(self._status_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Status saved to {self._status_file}")
        except Exception as e:
            logger.warning(f"Could not save status file: {e}")
    
    def _load_training_history(self) -> List[Dict[str, Any]]:
        """Load the full training history."""
        history_file = Path(self.config.backup_dir) / "training_history.json"
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load training history: {e}")
        return []
    
    def _save_training_history(self, history_entry: RetrainingHistory):
        """Append a new entry to training history."""
        history_file = Path(self.config.backup_dir) / "training_history.json"
        try:
            history = self._load_training_history()
            history.append(history_entry.to_dict())
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Training history saved: {len(history)} total entries")
        except Exception as e:
            logger.warning(f"Could not save training history: {e}")
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get the complete training history."""
        return self._load_training_history()
    
    def _update_image_counts(self):
        """Update counts of feedback images per class."""
        images_dir = Path(self.config.feedback_images_dir)
        
        counts = {name: 0 for name in self.CLASS_NAMES}
        junk_count = 0
        total = 0
        
        # Count corrected images
        corrected_dir = images_dir / "corrected"
        if corrected_dir.exists():
            for class_dir in corrected_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                    if class_name in counts:
                        counts[class_name] += count
                        total += count
        
        # Count confirmed correct images
        correct_dir = images_dir / "correct"
        if correct_dir.exists():
            for class_dir in correct_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                    if class_name in counts:
                        counts[class_name] += count
                        total += count
        
        # Count junk images
        junk_dir = images_dir / "junk"
        if junk_dir.exists():
            junk_count = len(list(junk_dir.glob("*.jpg"))) + len(list(junk_dir.glob("*.png")))
            total += junk_count
        
        with self._lock:
            self._status.images_per_class = counts
            self._status.junk_images = junk_count
            self._status.total_feedback_images = total
        
        logger.debug(f"Image counts updated: {total} total, {junk_count} junk")
    
    def get_status(self) -> RetrainingStatus:
        """Get current retraining status."""
        self._update_image_counts()
        return self._status
    
    def get_total_historical_feedback_count(self) -> int:
        """
        Get total count of ALL feedback images ever collected.
        Includes: current feedback + all archived feedback from previous training cycles.
        
        Returns:
            Total count of all historical feedback images
        """
        total = 0
        
        # Current feedback images
        total += self._status.total_feedback_images
        
        # Archived images from previous training cycles
        archive_dir = Path(self.config.training_history_dir)
        if archive_dir.exists():
            for version_dir in archive_dir.iterdir():
                if version_dir.is_dir() and version_dir.name.startswith('v'):
                    for folder in ["correct", "corrected", "junk"]:
                        folder_path = version_dir / folder
                        if folder_path.exists():
                            if folder in ["correct", "corrected"]:
                                # Count images in class subfolders
                                for class_dir in folder_path.iterdir():
                                    if class_dir.is_dir():
                                        total += len(list(class_dir.glob("*.jpg")))
                                        total += len(list(class_dir.glob("*.png")))
                            else:
                                # Junk folder has images directly
                                total += len(list(folder_path.glob("*.jpg")))
                                total += len(list(folder_path.glob("*.png")))
        
        return total
    
    def check_and_trigger_retraining(self) -> bool:
        """
        Check if retraining thresholds are met and trigger if so.
        Also checks if comprehensive training should be triggered (1000+ total images).
        
        Returns:
            True if retraining was triggered, False otherwise
        """
        self._update_image_counts()
        
        with self._lock:
            if self._status.is_training:
                logger.info("Retraining already in progress")
                return False
        
        # Check comprehensive training threshold (1000 total historical images)
        total_historical = self.get_total_historical_feedback_count()
        if total_historical >= 1000:
            # Try to trigger comprehensive training
            try:
                from .comprehensive_trainer import check_comprehensive_training_trigger, get_comprehensive_training_status
                
                status = get_comprehensive_training_status()
                if not status.get('is_training') and not status.get('is_completed'):
                    logger.info(f"📊 Total historical feedback: {total_historical} images - checking comprehensive training...")
                    if check_comprehensive_training_trigger(total_historical):
                        logger.info("🚀 COMPREHENSIVE TRAINING TRIGGERED!")
                        return True
            except ImportError as e:
                logger.warning(f"Comprehensive trainer not available: {e}")
        
        # Check regular fine-tuning threshold
        with self._lock:
            if not self._status.ready_to_retrain:
                logger.debug("Thresholds not met for retraining")
                return False
        
        # Trigger regular fine-tuning
        return self.start_retraining()
    
    def start_retraining(self, force: bool = False) -> bool:
        """
        Start the retraining process.
        
        Args:
            force: Force retraining even if thresholds not met
            
        Returns:
            True if retraining started, False otherwise
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, cannot retrain")
            return False
        
        with self._lock:
            if self._status.is_training:
                logger.warning("Retraining already in progress")
                return False
            
            if not force and not self._status.ready_to_retrain:
                logger.warning("Thresholds not met and force=False")
                return False
            
            self._status.is_training = True
            self._status.error = None
            self._status.training_progress = 0.0
        
        # Start training in background thread
        self._training_thread = threading.Thread(
            target=self._run_retraining,
            daemon=True
        )
        self._training_thread.start()
        
        logger.info("Retraining started in background")
        return True
    
    def _run_retraining(self):
        """Run the actual retraining process (in background thread)."""
        import time
        start_time = time.time()
        
        # Snapshot current image counts for history
        images_used = self._status.total_feedback_images
        images_per_class_used = dict(self._status.images_per_class)
        junk_images_used = self._status.junk_images
        
        backup_path = None
        training_metrics = {}
        try:
            # Step 1: Backup current model
            logger.info("Step 1: Backing up current model...")
            backup_path = self._backup_model()
            if backup_path:
                self._status.last_backup_path = str(backup_path)
            
            # Step 2: Load model
            logger.info("Step 2: Loading model...")
            model, checkpoint = self._load_model()
            if model is None:
                raise Exception("Failed to load model")
            
            # Step 3: Prepare dataset
            logger.info("Step 3: Preparing dataset...")
            dataset = self._prepare_dataset()
            if len(dataset) == 0:
                raise Exception("No feedback images found for training")
            
            # Step 4: Fine-tune model (returns model AND training_metrics)
            logger.info("Step 4: Fine-tuning model (rotation-robust methodology)...")
            model, training_metrics = self._fine_tune_model(model, dataset)
            
            # Check if training collapsed
            if training_metrics.get('collapse_detected', False):
                raise Exception("Training collapsed - model rolled back to pre-training state")
            
            # Step 5: Save updated model with metrics
            logger.info("Step 5: Saving updated model...")
            self._save_model(model, checkpoint, training_metrics)
            
            # Step 6: Archive used images (CRITICAL for cyclical retraining)
            logger.info("Step 6: Archiving trained images...")
            self._archive_feedback_images()
            
            # Calculate training duration
            training_duration = time.time() - start_time
            
            # Update version and counts
            with self._lock:
                self._status.is_training = False
                self._status.last_trained = datetime.now().isoformat()
                self._status.training_progress = 1.0
                self._status.current_version += 1  # Increment version
                self._status.total_retrains += 1
            
            # Save status to disk for persistence
            self._save_status()
            
            # Record in training history (with dual metrics)
            history_entry = RetrainingHistory(
                version=self._status.current_version,
                timestamp=self._status.last_trained,
                images_used=images_used,
                images_per_class=images_per_class_used,
                junk_images_used=junk_images_used,
                epochs=training_metrics.get('epochs_trained', self.config.epochs),
                backup_path=str(backup_path) if backup_path else "",
                success=True,
                training_duration_seconds=training_duration,
                final_upright_accuracy=training_metrics.get('final_upright_acc', 0),
                final_rotation_accuracy=training_metrics.get('final_rotation_acc', 0),
                best_upright_accuracy=training_metrics.get('best_upright_acc', 0),
                best_rotation_accuracy=training_metrics.get('best_rotation_acc', 0),
                collapse_detected=training_metrics.get('collapse_detected', False),
                early_stopped=training_metrics.get('early_stopped', False),
            )
            self._save_training_history(history_entry)
            
            logger.info(f"Retraining completed successfully! Model is now version {self._status.current_version}")
            logger.info(f"  Best upright: {training_metrics.get('best_upright_acc', 0):.2f}%")
            logger.info(f"  Best rotation: {training_metrics.get('best_rotation_acc', 0):.2f}%")
            logger.info(f"  Total retrains: {self._status.total_retrains}, Duration: {training_duration:.1f}s")
            
            # Callback
            if self._on_retrain_complete:
                self._on_retrain_complete(success=True)
                
        except Exception as e:
            logger.exception(f"Retraining failed: {e}")
            training_duration = time.time() - start_time
            
            with self._lock:
                self._status.is_training = False
                self._status.error = str(e)
            
            # Record failed attempt in history (with dual metrics if available)
            history_entry = RetrainingHistory(
                version=self._status.current_version,
                timestamp=datetime.now().isoformat(),
                images_used=images_used,
                images_per_class=images_per_class_used,
                junk_images_used=junk_images_used,
                epochs=training_metrics.get('epochs_trained', 0),
                backup_path=str(backup_path) if backup_path else "",
                success=False,
                error=str(e),
                training_duration_seconds=training_duration,
                final_upright_accuracy=training_metrics.get('final_upright_acc', 0),
                final_rotation_accuracy=training_metrics.get('final_rotation_acc', 0),
                best_upright_accuracy=training_metrics.get('best_upright_acc', 0),
                best_rotation_accuracy=training_metrics.get('best_rotation_acc', 0),
                collapse_detected=training_metrics.get('collapse_detected', False),
                early_stopped=training_metrics.get('early_stopped', False),
            )
            self._save_training_history(history_entry)
            
            if self._on_retrain_complete:
                self._on_retrain_complete(success=False, error=str(e))
    
    def _archive_feedback_images(self):
        """
        Move trained feedback images to archive folder.
        This prevents the same images from triggering immediate re-training.
        Images are moved to: {history_dir}/v{version}/
        """
        version = self._status.current_version
        archive_dir = Path(self.config.training_history_dir) / f"v{version}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = Path(self.config.feedback_images_dir)
        archived_count = 0
        
        # Archive each folder type
        for folder_name in ["correct", "corrected", "junk"]:
            src_folder = images_dir / folder_name
            if not src_folder.exists():
                continue
            
            dst_folder = archive_dir / folder_name
            dst_folder.mkdir(parents=True, exist_ok=True)
            
            # Move class subfolders for correct/corrected
            if folder_name in ["correct", "corrected"]:
                for class_dir in src_folder.iterdir():
                    if class_dir.is_dir():
                        dst_class_dir = dst_folder / class_dir.name
                        dst_class_dir.mkdir(parents=True, exist_ok=True)
                        
                        for img_file in class_dir.glob("*"):
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                try:
                                    shutil.move(str(img_file), str(dst_class_dir / img_file.name))
                                    archived_count += 1
                                except Exception as e:
                                    logger.warning(f"Could not archive {img_file}: {e}")
            else:
                # Junk folder has images directly
                for img_file in src_folder.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        try:
                            shutil.move(str(img_file), str(dst_folder / img_file.name))
                            archived_count += 1
                        except Exception as e:
                            logger.warning(f"Could not archive {img_file}: {e}")
        
        logger.info(f"Archived {archived_count} feedback images to {archive_dir}")
        
        # Update counts (should now be 0)
        self._update_image_counts()
    
    def _backup_model(self) -> Optional[Path]:
        """Backup current model before retraining."""
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}, skipping backup")
            return None
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{model_path.stem}_backup_{timestamp}{model_path.suffix}"
        backup_path = Path(self.config.backup_dir) / backup_name
        
        # Copy model
        shutil.copy2(model_path, backup_path)
        logger.info(f"Model backed up to {backup_path}")
        
        return backup_path
    
    def _load_model(self):
        """Load the current model for fine-tuning."""
        import sys
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Add KnowledgeDistillation src to path for model class
        kd_src = Path("D:/KnowledgeDistillation/src")
        if kd_src.exists() and str(kd_src) not in sys.path:
            sys.path.insert(0, str(kd_src))
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Determine number of classes
        num_classes = len(self.CLASS_NAMES)
        if self.config.include_junk_class:
            num_classes += 1  # Add junk class
        
        # Try to import EnhancedStudentModel
        try:
            from enhanced_student_model import EnhancedStudentModel  # type: ignore
            
            # Get base channels from state dict
            stem_weight = state_dict.get('stem.0.weight')
            base_channels = stem_weight.shape[0] if stem_weight is not None else 48
            
            # Check if we need to expand classifier for new classes
            original_num_classes = 11  # Original model has 11 classes
            
            model = EnhancedStudentModel(
                num_classes=original_num_classes,
                base_channels=base_channels
            )
            model.load_state_dict(state_dict, strict=False)
            
            # If adding junk class, expand the classifier
            if self.config.include_junk_class and num_classes > original_num_classes:
                model = self._expand_classifier(model, num_classes)
            
            logger.info(f"Loaded EnhancedStudentModel with {num_classes} classes")
            
        except ImportError:
            logger.warning("EnhancedStudentModel not found, trying MobileNetV3")
            import torchvision.models as models
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            model.load_state_dict(state_dict, strict=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        return model, checkpoint
    
    def _expand_classifier(self, model, new_num_classes: int):
        """Expand classifier for additional classes (e.g., junk)."""
        # Get the classifier module
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            # Find the last Linear layer
            for i in range(len(model.classifier) - 1, -1, -1):
                if isinstance(model.classifier[i], nn.Linear):
                    old_linear = model.classifier[i]
                    in_features = old_linear.in_features
                    old_num_classes = old_linear.out_features
                    
                    # Create new classifier with more outputs
                    new_linear = nn.Linear(in_features, new_num_classes)
                    
                    # Copy old weights
                    with torch.no_grad():
                        new_linear.weight[:old_num_classes] = old_linear.weight
                        new_linear.bias[:old_num_classes] = old_linear.bias
                        # Initialize new class weights
                        nn.init.xavier_uniform_(new_linear.weight[old_num_classes:])
                        nn.init.zeros_(new_linear.bias[old_num_classes:])
                    
                    model.classifier[i] = new_linear
                    logger.info(f"Expanded classifier from {old_num_classes} to {new_num_classes} classes")
                    break
        
        return model
    
    def _prepare_dataset(self) -> FeedbackImageDataset:
        """Prepare dataset from feedback images."""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FeedbackImageDataset(
            images_dir=Path(self.config.feedback_images_dir),
            class_names=self.CLASS_NAMES,
            include_junk=self.config.include_junk_class,
            transform=transform
        )
        
        return dataset
    
    def _compute_class_weights(self, dataset: FeedbackImageDataset) -> torch.Tensor:
        """
        Compute class weights for balanced training.
        Inverse frequency weighting prevents bias toward majority classes.
        """
        # Count samples per class
        class_counts = [0] * len(dataset.class_names)
        for _, label in dataset.samples:
            class_counts[label] += 1
        
        # Compute inverse frequency weights
        total_samples = sum(class_counts)
        weights = []
        for count in class_counts:
            if count > 0:
                # Inverse frequency with smoothing
                weight = total_samples / (len(class_counts) * count)
            else:
                weight = 1.0  # Default weight for classes with no samples
            weights.append(weight)
        
        # Normalize weights so they average to 1.0
        avg_weight = sum(weights) / len(weights)
        weights = [w / avg_weight for w in weights]
        
        logger.info(f"Class weights computed: {dict(zip(dataset.class_names, [f'{w:.2f}' for w in weights]))}")
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _create_balanced_sampler(self, dataset: FeedbackImageDataset):
        """
        Create a weighted sampler for balanced batch sampling.
        Oversamples minority classes so each batch has balanced representation.
        """
        from torch.utils.data import WeightedRandomSampler
        
        # Count samples per class
        class_counts = [0] * len(dataset.class_names)
        for _, label in dataset.samples:
            class_counts[label] += 1
        
        # Compute sample weights (inverse of class frequency)
        sample_weights = []
        for _, label in dataset.samples:
            class_count = class_counts[label]
            weight = 1.0 / class_count if class_count > 0 else 1.0
            sample_weights.append(weight)
        
        # Number of samples per epoch (oversample to largest class size * num_classes)
        num_samples = max(class_counts) * len([c for c in class_counts if c > 0])
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True  # Allow oversampling
        )
        
        logger.info(f"Balanced sampler: {num_samples} samples/epoch from {len(dataset)} original samples")
        
        return sampler
    
    def _compute_fisher_information(self, model, dataloader, device):
        """
        Compute Fisher Information Matrix for EWC.
        This captures which parameters are important for the current knowledge.
        """
        fisher_info = {}
        optimal_params = {}
        
        # Store current optimal parameters
        for name, param in model.named_parameters():
            optimal_params[name] = param.data.clone()
            fisher_info[name] = torch.zeros_like(param.data)
        
        model.eval()
        num_samples = 0
        
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            model.zero_grad()
            
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Use log-likelihood for Fisher computation
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            loss = torch.nn.functional.nll_loss(log_probs, labels)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
            
            num_samples += images.size(0)
        
        # Average over samples
        for name in fisher_info:
            fisher_info[name] /= num_samples
        
        logger.info(f"Fisher Information computed from {num_samples} samples")
        
        return fisher_info, optimal_params
    
    def _ewc_loss(self, model, fisher_info, optimal_params, ewc_lambda: float, max_loss: float) -> torch.Tensor:
        """
        Compute EWC penalty loss WITH CLAMPING.
        Penalizes changes to parameters that were important for previous tasks.
        Clamped to prevent EWC from dominating training (matching rotation script).
        """
        ewc_loss = 0.0
        
        for name, param in model.named_parameters():
            if name in fisher_info:
                # Quadratic penalty weighted by Fisher information
                ewc_loss += (fisher_info[name] * (param - optimal_params[name]) ** 2).sum()
        
        raw_loss = ewc_lambda * ewc_loss
        # CRITICAL: Clamp to prevent explosion (matching rotation training)
        clamped_loss = torch.clamp(raw_loss, max=max_loss)
        return clamped_loss
    
    def _apply_rotation(self, img: torch.Tensor, angle: int) -> torch.Tensor:
        """Apply rotation to a batch of images."""
        if angle == 0:
            return img
        elif angle == 90:
            return torch.rot90(img, k=1, dims=[2, 3])
        elif angle == 180:
            return torch.rot90(img, k=2, dims=[2, 3])
        elif angle == 270:
            return torch.rot90(img, k=3, dims=[2, 3])
        return img
    
    def _validate_upright(self, model, dataloader, device, criterion) -> Tuple[float, float]:
        """Validate on upright (non-rotated) images."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0
        return avg_loss, accuracy
    
    def _validate_rotation(self, model, dataloader, device) -> float:
        """
        Validate on rotated images (90°, 180°, 270°).
        Returns AVERAGE accuracy across rotations (matching rotation training script).
        """
        model.eval()
        rotation_angles = [90, 180, 270]
        accuracies = []
        
        with torch.no_grad():
            for angle in rotation_angles:
                correct = 0
                total = 0
                
                for images, labels in dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Apply rotation
                    rotated = self._apply_rotation(images, angle)
                    
                    outputs = model(rotated)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                acc = 100.0 * correct / total if total > 0 else 0
                accuracies.append(acc)
        
        avg_rotation_acc = sum(accuracies) / len(accuracies)
        return avg_rotation_acc
    
    def _fine_tune_model(self, model, dataset: FeedbackImageDataset) -> Dict[str, Any]:
        """
        Fine-tune model with feedback images using ROTATION-ROBUST methodology.
        
        MATCHES finetune_rotation_robust.py:
        1. Class-weighted loss (handles imbalanced data)
        2. Balanced sampling (oversamples minority classes)
        3. EWC with max_loss clamping (prevents forgetting AND explosion)
        4. Gradient clipping (prevents exploding gradients)
        5. Warmup epochs + cosine LR schedule
        6. DUAL-METRIC tracking (upright + rotation accuracy)
        7. Collapse detection with rollback
        8. Early stopping when both metrics stall
        9. Mixed precision training
        10. Progressive unfreezing
        """
        from torch.amp import GradScaler, autocast
        import copy
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # Training metrics to return
        training_metrics = {
            'epochs_trained': 0,
            'best_upright_acc': 0.0,
            'best_rotation_acc': 0.0,
            'final_upright_acc': 0.0,
            'final_rotation_acc': 0.0,
            'collapse_detected': False,
            'early_stopped': False,
            'history': []
        }
        
        # Save pre-training state for collapse recovery
        pre_training_state = copy.deepcopy(model.state_dict())
        
        # === STEP 1: Compute class weights for balanced loss ===
        class_weights = self._compute_class_weights(dataset).to(device)
        
        # === STEP 2: Create balanced dataloader ===
        if self.config.oversample_minority:
            sampler = self._create_balanced_sampler(dataset)
            train_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=True if device == 'cuda' else False
            )
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if device == 'cuda' else False
            )
        
        # Validation loader (no augmentation, no sampling)
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = FeedbackImageDataset(
            images_dir=Path(self.config.feedback_images_dir),
            class_names=self.CLASS_NAMES,
            include_junk=self.config.include_junk_class,
            transform=val_transform
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        
        # === STEP 3: Compute EWC Fisher information ===
        fisher_info = None
        optimal_params = None
        if self.config.use_ewc and len(dataset) > 0:
            logger.info("Computing Fisher Information for EWC...")
            fisher_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
            fisher_info, optimal_params = self._compute_fisher_information(model, fisher_loader, device)
        
        # === STEP 4: Setup optimizer and scheduler ===
        model.train()
        
        # Initially freeze backbone
        frozen_params = set()
        if self.config.freeze_backbone_epochs > 0:
            for name, param in model.named_parameters():
                if 'classifier' not in name and 'head' not in name:
                    param.requires_grad = False
                    frozen_params.add(name)
            logger.info(f"Backbone frozen for first {self.config.freeze_backbone_epochs} epochs")
        
        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Cosine annealing scheduler
        total_epochs = self.config.epochs
        if self.config.use_cosine_schedule:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - self.config.warmup_epochs,
                eta_min=self.config.min_learning_rate
            )
        else:
            scheduler = None
        
        # Loss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights if self.config.use_class_weights else None)
        
        # Mixed precision
        scaler = GradScaler('cuda') if self.config.use_amp and device == 'cuda' else None
        
        # === STEP 5: Training loop with dual-metric tracking ===
        self._status.total_epochs = total_epochs
        best_upright_acc = 0.0
        best_rotation_acc = 0.0
        best_state = None
        epochs_no_upright_improve = 0
        epochs_no_rotation_improve = 0
        prev_train_acc = None
        
        for epoch in range(total_epochs):
            self._status.current_epoch = epoch + 1
            
            # === Warmup learning rate ===
            if epoch < self.config.warmup_epochs:
                warmup_lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                logger.info(f"  [Warmup] epoch {epoch+1}/{self.config.warmup_epochs}, LR: {warmup_lr:.6f}")
            
            # === Progressive unfreezing ===
            if epoch == self.config.freeze_backbone_epochs and frozen_params:
                logger.info("Unfreezing backbone layers...")
                for name, param in model.named_parameters():
                    if name in frozen_params:
                        param.requires_grad = True
                
                # Re-create optimizer with all parameters and lower LR for backbone
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate * 0.1,
                    weight_decay=self.config.weight_decay
                )
                if self.config.use_cosine_schedule:
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=total_epochs - epoch,
                        eta_min=self.config.min_learning_rate
                    )
            
            # === Train one epoch ===
            model.train()
            epoch_loss = 0.0
            ewc_loss_total = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if scaler is not None:
                    with autocast('cuda'):
                        outputs = model(images)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        ce_loss = criterion(logits, labels)
                        
                        # Add EWC loss with clamping
                        if fisher_info is not None and self.config.use_ewc:
                            ewc_penalty = self._ewc_loss(
                                model, fisher_info, optimal_params,
                                self.config.ewc_lambda, self.config.max_ewc_loss
                            )
                            loss = ce_loss + ewc_penalty
                            ewc_loss_total += ewc_penalty.item()
                        else:
                            loss = ce_loss
                    
                    # Backward with gradient clipping
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.config.gradient_clip_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Non-AMP training
                    outputs = model(images)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    ce_loss = criterion(logits, labels)
                    
                    if fisher_info is not None and self.config.use_ewc:
                        ewc_penalty = self._ewc_loss(
                            model, fisher_info, optimal_params,
                            self.config.ewc_lambda, self.config.max_ewc_loss
                        )
                        loss = ce_loss + ewc_penalty
                        ewc_loss_total += ewc_penalty.item()
                    else:
                        loss = ce_loss
                    
                    loss.backward()
                    # CRITICAL: Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.config.gradient_clip_norm
                    )
                    optimizer.step()
                
                epoch_loss += ce_loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # Update scheduler (after warmup)
            if scheduler is not None and epoch >= self.config.warmup_epochs:
                scheduler.step()
            
            train_acc = 100.0 * correct / total if total > 0 else 0
            avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
            avg_ewc = ewc_loss_total / len(train_loader) if len(train_loader) > 0 else 0
            
            # === COLLAPSE DETECTION ===
            if prev_train_acc is not None and (prev_train_acc - train_acc) > self.config.collapse_threshold:
                logger.warning(f"⚠️ COLLAPSE DETECTED! Accuracy dropped from {prev_train_acc:.2f}% to {train_acc:.2f}%")
                logger.warning("Restoring pre-training state and aborting...")
                model.load_state_dict(pre_training_state)
                training_metrics['collapse_detected'] = True
                training_metrics['epochs_trained'] = epoch
                return model, training_metrics
            
            prev_train_acc = train_acc
            
            # === DUAL VALIDATION ===
            val_loss, upright_acc = self._validate_upright(model, val_loader, device, criterion)
            
            if self.config.validate_rotation:
                rotation_acc = self._validate_rotation(model, val_loader, device)
            else:
                rotation_acc = upright_acc
            
            # Track improvements
            if upright_acc > best_upright_acc + self.config.min_accuracy_improvement:
                best_upright_acc = upright_acc
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_upright_improve = 0
            else:
                epochs_no_upright_improve += 1
            
            if rotation_acc > best_rotation_acc + self.config.min_accuracy_improvement:
                best_rotation_acc = rotation_acc
                epochs_no_rotation_improve = 0
            else:
                epochs_no_rotation_improve += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            progress = (epoch + 1) / total_epochs
            self._status.training_progress = progress * 0.9
            
            logger.info(
                f"Epoch {epoch+1}/{total_epochs} - "
                f"Train: {train_acc:.2f}% | "
                f"Upright: {upright_acc:.2f}% (best: {best_upright_acc:.2f}%, stall: {epochs_no_upright_improve}/{self.config.patience}) | "
                f"Rotation: {rotation_acc:.2f}% (best: {best_rotation_acc:.2f}%, stall: {epochs_no_rotation_improve}/{self.config.patience}) | "
                f"EWC: {avg_ewc:.4f} | LR: {current_lr:.6f}"
            )
            
            # Record history
            training_metrics['history'].append({
                'epoch': epoch + 1,
                'train_acc': train_acc,
                'upright_acc': upright_acc,
                'rotation_acc': rotation_acc,
                'loss': avg_loss,
                'ewc_loss': avg_ewc,
                'lr': current_lr
            })
            
            # === DUAL-METRIC EARLY STOPPING ===
            upright_stalled = epochs_no_upright_improve >= self.config.patience
            rotation_stalled = epochs_no_rotation_improve >= self.config.patience
            
            if upright_stalled and rotation_stalled:
                logger.info(f"★ DUAL-METRIC early stopping at epoch {epoch+1}")
                logger.info(f"  Upright stalled for {epochs_no_upright_improve} epochs")
                logger.info(f"  Rotation stalled for {epochs_no_rotation_improve} epochs")
                training_metrics['early_stopped'] = True
                break
        
        # Restore best state if we have one
        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info(f"Restored best model state (upright: {best_upright_acc:.2f}%, rotation: {best_rotation_acc:.2f}%)")
        
        # Final metrics
        training_metrics['epochs_trained'] = epoch + 1
        training_metrics['best_upright_acc'] = best_upright_acc
        training_metrics['best_rotation_acc'] = best_rotation_acc
        training_metrics['final_upright_acc'] = upright_acc
        training_metrics['final_rotation_acc'] = rotation_acc
        
        logger.info(f"Training complete. Best upright: {best_upright_acc:.2f}%, Best rotation: {best_rotation_acc:.2f}%")
        
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        model.eval()
        return model, training_metrics
    
    def _save_model(self, model, original_checkpoint: dict, training_metrics: Optional[Dict] = None):
        """Save the fine-tuned model with training metrics."""
        model_path = Path(self.config.model_path)
        
        # Create new checkpoint with version info
        new_checkpoint = {
            'model_state_dict': model.state_dict(),
            'class_names': self.CLASS_NAMES + (['junk'] if self.config.include_junk_class else []),
            'fine_tuned': True,
            'fine_tune_date': datetime.now().isoformat(),
            'feedback_images_count': self._status.total_feedback_images,
            # Version tracking
            'model_version': self._status.current_version + 1,  # +1 because version increments after save
            'total_retrains': self._status.total_retrains + 1,
        }
        
        # Add training metrics if available
        if training_metrics:
            new_checkpoint['training_metrics'] = {
                'best_upright_accuracy': training_metrics.get('best_upright_acc', 0),
                'best_rotation_accuracy': training_metrics.get('best_rotation_acc', 0),
                'epochs_trained': training_metrics.get('epochs_trained', 0),
                'early_stopped': training_metrics.get('early_stopped', False),
            }
        
        # Preserve some original checkpoint info
        for key in ['training_config', 'original_training_date']:
            if key in original_checkpoint:
                new_checkpoint[key] = original_checkpoint[key]
        
        # Save
        torch.save(new_checkpoint, model_path)
        logger.info(f"Model saved to {model_path} (will be version {new_checkpoint['model_version']})")
        
        self._status.training_progress = 1.0
    
    def set_on_retrain_complete(self, callback: Callable):
        """Set callback for when retraining completes."""
        self._on_retrain_complete = callback
    
    def record_feedback_image(
        self,
        class_name: str,
        image_path: str,
        is_correction: bool = True,
        is_junk: bool = False
    ):
        """
        Record a new feedback image and check if retraining should trigger.
        
        Args:
            class_name: The correct class for this image
            image_path: Path where image was saved
            is_correction: True if this corrects a wrong prediction
            is_junk: True if image is reported as junk/unrelated
        """
        # Update counts
        self._update_image_counts()
        
        # Log
        logger.debug(f"Feedback image recorded: {class_name} (junk={is_junk})")
        
        # Check if we should trigger retraining
        if self._status.ready_to_retrain and not self._status.is_training:
            logger.info("Retraining threshold reached! Triggering auto-retrain...")
            self.start_retraining()


# Global instance
_retrain_manager: Optional[ModelRetrainingManager] = None


def get_retrain_manager(
    model_path: Optional[str] = None,
    feedback_dir: Optional[str] = None,
    **kwargs
) -> ModelRetrainingManager:
    """Get or create the global retraining manager."""
    global _retrain_manager
    if _retrain_manager is None:
        config = RetrainingConfig(**kwargs)
        _retrain_manager = ModelRetrainingManager(
            config=config,
            model_path=model_path,
            feedback_dir=feedback_dir,
        )
    return _retrain_manager


def check_retrain_status() -> Dict[str, Any]:
    """Get current retraining status including comprehensive training."""
    if _retrain_manager is None:
        return {"error": "Retraining manager not initialized"}
    
    status = _retrain_manager.get_status().to_dict()
    
    # Add total historical count
    status["total_historical_feedback"] = _retrain_manager.get_total_historical_feedback_count()
    status["comprehensive_threshold"] = 1000
    status["comprehensive_triggered"] = status["total_historical_feedback"] >= 1000
    
    # Add comprehensive training status if available
    try:
        from .comprehensive_trainer import get_comprehensive_training_status
        status["comprehensive_training"] = get_comprehensive_training_status()
    except ImportError:
        status["comprehensive_training"] = None
    
    return status
