"""
Comprehensive Training Script - Full 360° Rotation Robust Model
================================================================
Triggers when total feedback images reach 1000.
Uses ALL available data sources:
- Archived feedback images (from previous retraining cycles)
- Original datasets (D:\Test-images, G:\AI work\IMAGE DATASET)
- Current feedback images

Features (matching finetune_rotation_robust.py):
- Full 360° rotation augmentation (0°, 15°, 30°, 45°, 60°, 75°, 90°, ... 345°)
- Dual-metric tracking (upright + rotation accuracy)
- EWC with max_loss clamping (prevents forgetting & explosion)
- Gradient clipping (max_norm=1.0)
- Collapse detection (>30% drop) with rollback
- Warmup epochs + Cosine LR schedule
- Mixed precision (AMP)
- Progressive unfreezing
- Dual-metric early stopping
- Class-weighted loss
- Balanced sampling
- CHECKPOINT RECOVERY (survives power outages)

Author: Intelli-PEST Backend Team
Date: 2026-01-06
"""

import os
import sys
import json
import copy
import random
import shutil
import logging
import threading
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy imports
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, ConcatDataset
    from torch.amp import GradScaler, autocast
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available")

PIL_AVAILABLE = False
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("PIL not available")

import numpy as np


# ============================================================
# Configuration
# ============================================================

@dataclass
class ComprehensiveTrainingConfig:
    """Configuration for comprehensive training."""
    
    # Trigger threshold
    trigger_threshold: int = 1000  # Total feedback images to trigger
    
    # Data sources
    dataset_paths: List[str] = field(default_factory=lambda: [
        "D:/Test-images",
        "D:/IMAGE DATASET",
    ])
    feedback_images_dir: str = "./feedback_data/images"
    archived_images_dir: str = "./model_backups/history"
    
    # Model paths
    model_path: str = "D:/KnowledgeDistillation/student_model_rotation_robust.pt"
    backup_dir: str = "./model_backups"
    checkpoint_dir: str = "./model_backups/comprehensive_checkpoints"
    
    # Training parameters
    epochs: int = 50  # More epochs for comprehensive training
    batch_size: int = 16
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    
    # Knowledge Distillation parameters
    use_knowledge_distillation: bool = True  # Use all 11 teachers + deployed model
    kd_temperature: float = 4.0  # Temperature for soft labels (higher = softer)
    kd_alpha: float = 0.3  # Hard label (ground truth) weight
    kd_beta: float = 0.7   # Soft label (teacher) weight
    teacher_models_dir: str = "D:/Intelli_PEST-Backend/tflite_models_compatible/onnx_models"
    use_student_as_teacher: bool = True  # Include deployed student as 12th teacher
    use_sequential_teachers: bool = True  # Train with ONE teacher at a time
    epochs_per_teacher: int = 25  # Epochs per teacher for sequential KD training
    
    # EWC (Elastic Weight Consolidation)
    use_ewc: bool = True
    ewc_lambda: float = 100.0
    max_ewc_loss: float = 1.0
    
    # Gradient clipping
    gradient_clip_norm: float = 1.0
    
    # Learning rate schedule
    warmup_epochs: int = 3
    use_cosine_schedule: bool = True
    
    # Early stopping (dual-metric)
    patience: int = 8  # More patience for comprehensive training
    min_accuracy_improvement: float = 0.1
    
    # Collapse detection
    collapse_threshold: float = 30.0
    
    # Rotation augmentation - FULL 360°
    rotation_angles: List[int] = field(default_factory=lambda: [
        0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
        180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345
    ])
    cardinal_rotation_prob: float = 0.5  # 50% chance of cardinal (0, 90, 180, 270)
    minor_rotation_prob: float = 0.3     # 30% chance of minor angles
    
    # Class balancing
    use_class_weights: bool = True
    oversample_minority: bool = True
    
    # Mixed precision
    use_amp: bool = True
    
    # Progressive unfreezing
    freeze_backbone_epochs: int = 5
    
    # Checkpointing (power outage recovery)
    checkpoint_every_n_epochs: int = 2
    save_best_model: bool = True
    
    # Class names
    class_names: List[str] = field(default_factory=lambda: [
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
    ])
    include_junk_class: bool = True


@dataclass
class ComprehensiveTrainingState:
    """
    Persistent training state for checkpoint recovery.
    Survives power outages and server restarts.
    """
    # Training progress
    current_epoch: int = 0
    total_epochs: int = 50
    is_training: bool = False
    is_completed: bool = False
    
    # Best metrics
    best_upright_accuracy: float = 0.0
    best_rotation_accuracy: float = 0.0
    best_epoch: int = 0
    
    # Current metrics
    current_upright_accuracy: float = 0.0
    current_rotation_accuracy: float = 0.0
    
    # Early stopping
    epochs_no_upright_improve: int = 0
    epochs_no_rotation_improve: int = 0
    
    # Timestamps
    started_at: Optional[str] = None
    last_checkpoint_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Data info
    total_images_used: int = 0
    images_per_class: Dict[str, int] = field(default_factory=dict)
    
    # Version
    training_version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComprehensiveTrainingState':
        return cls(**data)


# ============================================================
# Full 360° Rotation Augmentation
# ============================================================

class Full360RotationAugmentation:
    """
    Apply full 360° rotation augmentation.
    Includes both cardinal (0°, 90°, 180°, 270°) and minor angles.
    """
    
    def __init__(
        self,
        cardinal_prob: float = 0.5,
        minor_prob: float = 0.3,
        cardinal_angles: List[int] = None,
        minor_angles: List[int] = None
    ):
        self.cardinal_prob = cardinal_prob
        self.minor_prob = minor_prob
        self.cardinal_angles = cardinal_angles or [0, 90, 180, 270]
        self.minor_angles = minor_angles or [15, 30, 45, 60, 75, 105, 120, 135, 150, 165,
                                              195, 210, 225, 240, 255, 285, 300, 315, 330, 345]
    
    def __call__(self, img: Image.Image) -> Tuple[Image.Image, int]:
        """
        Apply rotation augmentation.
        
        Returns:
            img: Rotated image
            angle: Applied rotation angle (0 if no rotation)
        """
        rand = random.random()
        
        if rand < self.cardinal_prob:
            # Apply cardinal rotation
            angle = random.choice(self.cardinal_angles)
        elif rand < self.cardinal_prob + self.minor_prob:
            # Apply minor rotation
            angle = random.choice(self.minor_angles)
        else:
            # No rotation
            angle = 0
        
        if angle != 0:
            # Use BILINEAR for quality
            img = img.rotate(-angle, expand=True, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        
        return img, angle


# ============================================================
# Combined Dataset
# ============================================================

class ComprehensiveDataset(Dataset):
    """
    Dataset combining all data sources:
    - Original datasets
    - Archived feedback images
    - Current feedback images
    
    With full 360° rotation augmentation.
    """
    
    def __init__(
        self,
        data_paths: List[str],
        class_names: List[str],
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        image_size: int = 256,
        rotation_augmentation: Full360RotationAugmentation = None,
        is_training: bool = True
    ):
        self.data_paths = [Path(p) for p in data_paths if Path(p).exists()]
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.split = split
        self.image_size = image_size
        self.rotation_aug = rotation_augmentation if is_training else None
        self.is_training = is_training
        
        # Transforms
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Load samples
        self.samples = self._load_and_split_samples(train_ratio, seed)
        
        logger.info(f"ComprehensiveDataset [{split}]: {len(self.samples)} samples from {len(self.data_paths)} sources")
    
    def _load_and_split_samples(self, train_ratio: float, seed: int) -> List[Tuple[str, int]]:
        """Load samples from all data sources."""
        all_samples = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for data_path in self.data_paths:
            if not data_path.exists():
                continue
            
            # Check if it's a feedback-style directory (with correct/corrected/junk subfolders)
            if (data_path / "correct").exists() or (data_path / "corrected").exists():
                # Feedback images structure
                for folder_type in ["correct", "corrected"]:
                    folder = data_path / folder_type
                    if folder.exists():
                        for class_dir in folder.iterdir():
                            if class_dir.is_dir():
                                class_name = class_dir.name
                                if class_name in self.class_to_idx:
                                    class_idx = self.class_to_idx[class_name]
                                    for img_path in class_dir.iterdir():
                                        if img_path.suffix.lower() in valid_extensions:
                                            all_samples.append((str(img_path), class_idx))
                
                # Junk images
                junk_folder = data_path / "junk"
                if junk_folder.exists() and "junk" in self.class_to_idx:
                    junk_idx = self.class_to_idx["junk"]
                    for img_path in junk_folder.iterdir():
                        if img_path.suffix.lower() in valid_extensions:
                            all_samples.append((str(img_path), junk_idx))
            else:
                # Standard dataset structure (class folders directly)
                for class_dir in data_path.iterdir():
                    if class_dir.is_dir() and not class_dir.name.startswith('.'):
                        class_name = class_dir.name
                        if class_name in self.class_to_idx:
                            class_idx = self.class_to_idx[class_name]
                            for img_path in class_dir.iterdir():
                                if img_path.suffix.lower() in valid_extensions:
                                    all_samples.append((str(img_path), class_idx))
        
        # Split by class
        np.random.seed(seed)
        class_samples = {}
        for path, label in all_samples:
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append((path, label))
        
        train_samples, val_samples = [], []
        for label, samples in class_samples.items():
            np.random.shuffle(samples)
            split_idx = int(len(samples) * train_ratio)
            train_samples.extend(samples[:split_idx])
            val_samples.extend(samples[split_idx:])
        
        result = train_samples if self.split == "train" else val_samples
        np.random.shuffle(result)
        return result
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
            img: Image tensor
            label: Class label
            rotation_angle: Applied rotation (0 if none)
        """
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return placeholder
            return torch.zeros(3, self.image_size, self.image_size), label, 0
        
        # Apply rotation augmentation
        rotation_angle = 0
        if self.rotation_aug:
            img, rotation_angle = self.rotation_aug(img)
        
        # Apply other transforms
        img = self.transform(img)
        
        return img, label, rotation_angle
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get counts per class."""
        counts = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            class_name = self.class_names[label]
            counts[class_name] += 1
        return counts


# ============================================================
# Comprehensive Trainer
# ============================================================

class ComprehensiveTrainer:
    """
    Full comprehensive training with all features.
    Includes checkpoint recovery for power outage survival.
    """
    
    def __init__(self, config: ComprehensiveTrainingConfig = None):
        self.config = config or ComprehensiveTrainingConfig()
        
        # Create directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)
        
        # State file for recovery
        self._state_file = Path(self.config.checkpoint_dir) / "training_state.json"
        self._checkpoint_file = Path(self.config.checkpoint_dir) / "latest_checkpoint.pt"
        self._best_model_file = Path(self.config.checkpoint_dir) / "best_model.pt"
        
        # Load or create state
        self._state = self._load_state()
        
        # Training components (lazy init)
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
        self._fisher_info = None
        self._optimal_params = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
        self._training_thread = None
        
        logger.info("ComprehensiveTrainer initialized")
        logger.info(f"  Checkpoint dir: {self.config.checkpoint_dir}")
        logger.info(f"  State file: {self._state_file}")
    
    def _load_state(self) -> ComprehensiveTrainingState:
        """Load training state from disk."""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    data = json.load(f)
                    state = ComprehensiveTrainingState.from_dict(data)
                    logger.info(f"Loaded training state: epoch {state.current_epoch}/{state.total_epochs}")
                    return state
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
        return ComprehensiveTrainingState()
    
    def _save_state(self):
        """Save training state to disk."""
        try:
            self._state.last_checkpoint_at = datetime.now().isoformat()
            with open(self._state_file, 'w') as f:
                json.dump(self._state.to_dict(), f, indent=2)
            logger.debug(f"State saved to {self._state_file}")
        except Exception as e:
            logger.warning(f"Could not save state: {e}")
    
    def _save_checkpoint(self, epoch: int, model, optimizer, scheduler, is_best: bool = False):
        """Save training checkpoint for recovery."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_upright_accuracy': self._state.best_upright_accuracy,
            'best_rotation_accuracy': self._state.best_rotation_accuracy,
            'fisher_info': self._fisher_info,
            'optimal_params': self._optimal_params,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self._checkpoint_file)
        logger.info(f"Checkpoint saved at epoch {epoch}")
        
        # Save best model if applicable
        if is_best:
            torch.save(checkpoint, self._best_model_file)
            logger.info(f"Best model saved (upright: {self._state.best_upright_accuracy:.2f}%)")
    
    def _load_checkpoint(self, model, optimizer, scheduler):
        """Load checkpoint for resuming training."""
        if not self._checkpoint_file.exists():
            return False
        
        try:
            checkpoint = torch.load(self._checkpoint_file, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self._fisher_info = checkpoint.get('fisher_info')
            self._optimal_params = checkpoint.get('optimal_params')
            
            logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
            return True
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return False
    
    def get_state(self) -> ComprehensiveTrainingState:
        """Get current training state."""
        return self._state
    
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._state.is_training
    
    def can_resume(self) -> bool:
        """Check if training can be resumed from checkpoint."""
        return (
            self._checkpoint_file.exists() and 
            not self._state.is_completed and 
            self._state.current_epoch > 0
        )
    
    def start_training(self, resume: bool = True) -> bool:
        """
        Start comprehensive training.
        
        Args:
            resume: If True, resume from checkpoint if available
            
        Returns:
            True if training started, False otherwise
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available")
            return False
        
        with self._lock:
            if self._state.is_training:
                logger.warning("Training already in progress")
                return False
            
            self._state.is_training = True
            self._state.is_completed = False
            if not resume or self._state.current_epoch == 0:
                self._state.started_at = datetime.now().isoformat()
                self._state.current_epoch = 0
            self._save_state()
        
        # Start training in background thread
        self._training_thread = threading.Thread(
            target=self._run_training,
            args=(resume,),
            daemon=True
        )
        self._training_thread.start()
        
        logger.info("Comprehensive training started in background")
        return True
    
    def _run_training(self, resume: bool):
        """Run the comprehensive training process."""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Training on device: {device}")
            
            # === STEP 1: Collect all data sources ===
            logger.info("Step 1: Collecting data sources...")
            data_paths = self._collect_data_paths()
            logger.info(f"Found {len(data_paths)} data sources")
            
            # === STEP 2: Create datasets ===
            logger.info("Step 2: Creating datasets...")
            class_names = list(self.config.class_names)
            if self.config.include_junk_class and "junk" not in class_names:
                class_names.append("junk")
            
            rotation_aug = Full360RotationAugmentation(
                cardinal_prob=self.config.cardinal_rotation_prob,
                minor_prob=self.config.minor_rotation_prob,
                cardinal_angles=[0, 90, 180, 270],
                minor_angles=[a for a in self.config.rotation_angles if a not in [0, 90, 180, 270]]
            )
            
            train_dataset = ComprehensiveDataset(
                data_paths=data_paths,
                class_names=class_names,
                split="train",
                rotation_augmentation=rotation_aug,
                is_training=True
            )
            
            val_dataset = ComprehensiveDataset(
                data_paths=data_paths,
                class_names=class_names,
                split="val",
                is_training=False
            )
            
            logger.info(f"Train dataset: {len(train_dataset)} samples")
            logger.info(f"Val dataset: {len(val_dataset)} samples")
            
            # Update state
            self._state.total_images_used = len(train_dataset) + len(val_dataset)
            self._state.images_per_class = train_dataset.get_class_counts()
            self._save_state()
            
            # === STEP 3: Create dataloaders ===
            logger.info("Step 3: Creating dataloaders...")
            train_loader = self._create_balanced_dataloader(train_dataset, device)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=12, persistent_workers=True)
            
            # === STEP 4: Load model ===
            logger.info("Step 4: Loading model...")
            model = self._load_model(len(class_names), device)
            
            # Backup original model before training
            self._backup_model()
            
            # === STEP 5: Setup training components ===
            logger.info("Step 5: Setting up training components...")
            optimizer, scheduler, criterion, scaler = self._setup_training(model, train_dataset, device)
            
            # === STEP 6: Resume from checkpoint if available ===
            start_epoch = 0
            if resume and self.can_resume():
                logger.info("Resuming from checkpoint...")
                if self._load_checkpoint(model, optimizer, scheduler):
                    start_epoch = self._state.current_epoch
                    logger.info(f"Resuming from epoch {start_epoch}")
            
            # Compute EWC if not already done
            if self.config.use_ewc and self._fisher_info is None:
                logger.info("Computing Fisher Information for EWC...")
                self._compute_fisher_information(model, train_loader, device)
            
            # === STEP 7: Training loop ===
            logger.info("Step 7: Starting training loop...")
            
            pre_training_state = copy.deepcopy(model.state_dict())
            best_state = None
            prev_train_acc = None
            
            # Check if we have sequential KD
            sequential_kd = getattr(self, '_sequential_kd', None)
            kd_loss_fn = getattr(self, '_kd_loss_fn', None)
            
            if sequential_kd is not None and kd_loss_fn is not None:
                # === SEQUENTIAL TEACHER TRAINING ===
                total_teachers = sequential_kd.get_teacher_count()
                total_epochs = total_teachers * self.config.epochs_per_teacher
                self._state.total_epochs = total_epochs
                global_epoch = 0
                
                logger.info(f"Sequential KD Training: {total_teachers} teachers × {self.config.epochs_per_teacher} epochs = {total_epochs} total epochs")
                
                for teacher_name, teacher, phase_num, total_phases in sequential_kd.iterate_teachers():
                    logger.info(f"═══════════════════════════════════════════")
                    logger.info(f"Training with teacher: {teacher_name} ({phase_num}/{total_phases})")
                    logger.info(f"═══════════════════════════════════════════")
                    
                    for local_epoch in range(self.config.epochs_per_teacher):
                        global_epoch += 1
                        self._state.current_epoch = global_epoch
                        
                        # === Warmup (first teacher only) ===
                        if phase_num == 1 and local_epoch < self.config.warmup_epochs:
                            warmup_lr = self.config.learning_rate * (local_epoch + 1) / self.config.warmup_epochs
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = warmup_lr
                            logger.info(f"[Warmup] LR: {warmup_lr:.6f}")
                        
                        # === Progressive unfreezing ===
                        if global_epoch == self.config.freeze_backbone_epochs:
                            self._unfreeze_backbone(model, optimizer)
                        
                        # === Train epoch with SINGLE teacher ===
                        train_loss, train_acc = self._train_epoch_with_teacher(
                            model, train_loader, optimizer, criterion, scaler, device, global_epoch, teacher, kd_loss_fn
                        )
                        
                        # === Collapse detection ===
                        if prev_train_acc is not None and (prev_train_acc - train_acc) > self.config.collapse_threshold:
                            logger.warning(f"⚠️ COLLAPSE DETECTED! Accuracy dropped {prev_train_acc:.2f}% → {train_acc:.2f}%")
                            logger.warning("Rolling back to pre-training state...")
                            model.load_state_dict(pre_training_state)
                            self._state.is_training = False
                            self._state.is_completed = False
                            self._save_state()
                            return
                        
                        prev_train_acc = train_acc
                        
                        # === Update scheduler ===
                        if scheduler and global_epoch >= self.config.warmup_epochs:
                            scheduler.step()
                        
                        # === Dual validation ===
                        val_loss, upright_acc = self._validate_upright(model, val_loader, criterion, device)
                        rotation_acc = self._validate_rotation(model, val_loader, device)
                        
                        self._state.current_upright_accuracy = upright_acc
                        self._state.current_rotation_accuracy = rotation_acc
                        
                        # Track improvements
                        is_best = False
                        if upright_acc > self._state.best_upright_accuracy + self.config.min_accuracy_improvement:
                            self._state.best_upright_accuracy = upright_acc
                            self._state.best_epoch = global_epoch
                            self._state.epochs_no_upright_improve = 0
                            best_state = copy.deepcopy(model.state_dict())
                            is_best = True
                        else:
                            self._state.epochs_no_upright_improve += 1
                        
                        if rotation_acc > self._state.best_rotation_accuracy + self.config.min_accuracy_improvement:
                            self._state.best_rotation_accuracy = rotation_acc
                            self._state.epochs_no_rotation_improve = 0
                        else:
                            self._state.epochs_no_rotation_improve += 1
                        
                        current_lr = optimizer.param_groups[0]['lr']
                        logger.info(
                            f"[{teacher_name}] Epoch {local_epoch+1}/{self.config.epochs_per_teacher} (Global: {global_epoch}/{total_epochs}) - "
                            f"Train: {train_acc:.2f}% | "
                            f"Upright: {upright_acc:.2f}% (best: {self._state.best_upright_accuracy:.2f}%) | "
                            f"Rotation: {rotation_acc:.2f}% (best: {self._state.best_rotation_accuracy:.2f}%)"
                        )
                        
                        # === Save checkpoint ===
                        if global_epoch % self.config.checkpoint_every_n_epochs == 0 or is_best:
                            self._save_checkpoint(global_epoch, model, optimizer, scheduler, is_best)
                        
                        self._save_state()
                
                # Sequential KD complete - restore best and finalize
                logger.info("Step 8: Finalizing...")
                if best_state is not None:
                    model.load_state_dict(best_state)
                    logger.info(f"Restored best model (upright: {self._state.best_upright_accuracy:.2f}%, rotation: {self._state.best_rotation_accuracy:.2f}%)")
                
                # Save final model
                self._save_final_model(model)
                self._state.is_training = False
                self._state.is_completed = True
                self._save_state()
                
                logger.info(f"Sequential KD Comprehensive training complete!")
                logger.info(f"Best upright accuracy: {self._state.best_upright_accuracy:.2f}%")
                logger.info(f"Best rotation accuracy: {self._state.best_rotation_accuracy:.2f}%")
                return
            
            # === FALLBACK: Original training loop (no sequential KD) ===
            self._state.total_epochs = self.config.epochs
            for epoch in range(start_epoch, self.config.epochs):
                self._state.current_epoch = epoch + 1
                
                # === Warmup ===
                if epoch < self.config.warmup_epochs:
                    warmup_lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warmup_lr
                    logger.info(f"[Warmup] Epoch {epoch+1}, LR: {warmup_lr:.6f}")
                
                # === Progressive unfreezing ===
                if epoch == self.config.freeze_backbone_epochs:
                    self._unfreeze_backbone(model, optimizer)
                
                # === Train epoch ===
                train_loss, train_acc = self._train_epoch(
                    model, train_loader, optimizer, criterion, scaler, device, epoch
                )
                
                # === Collapse detection ===
                if prev_train_acc is not None and (prev_train_acc - train_acc) > self.config.collapse_threshold:
                    logger.warning(f"⚠️ COLLAPSE DETECTED! Accuracy dropped {prev_train_acc:.2f}% → {train_acc:.2f}%")
                    logger.warning("Rolling back to pre-training state...")
                    model.load_state_dict(pre_training_state)
                    self._state.is_training = False
                    self._state.is_completed = False
                    self._save_state()
                    return
                
                prev_train_acc = train_acc
                
                # === Update scheduler ===
                if scheduler and epoch >= self.config.warmup_epochs:
                    scheduler.step()
                
                # === Dual validation ===
                val_loss, upright_acc = self._validate_upright(model, val_loader, criterion, device)
                rotation_acc = self._validate_rotation(model, val_loader, device)
                
                self._state.current_upright_accuracy = upright_acc
                self._state.current_rotation_accuracy = rotation_acc
                
                # Track improvements
                is_best = False
                if upright_acc > self._state.best_upright_accuracy + self.config.min_accuracy_improvement:
                    self._state.best_upright_accuracy = upright_acc
                    self._state.best_epoch = epoch + 1
                    self._state.epochs_no_upright_improve = 0
                    best_state = copy.deepcopy(model.state_dict())
                    is_best = True
                else:
                    self._state.epochs_no_upright_improve += 1
                
                if rotation_acc > self._state.best_rotation_accuracy + self.config.min_accuracy_improvement:
                    self._state.best_rotation_accuracy = rotation_acc
                    self._state.epochs_no_rotation_improve = 0
                else:
                    self._state.epochs_no_rotation_improve += 1
                
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train: {train_acc:.2f}% | "
                    f"Upright: {upright_acc:.2f}% (best: {self._state.best_upright_accuracy:.2f}%, stall: {self._state.epochs_no_upright_improve}/{self.config.patience}) | "
                    f"Rotation: {rotation_acc:.2f}% (best: {self._state.best_rotation_accuracy:.2f}%, stall: {self._state.epochs_no_rotation_improve}/{self.config.patience}) | "
                    f"LR: {current_lr:.6f}"
                )
                
                # === Save checkpoint ===
                if (epoch + 1) % self.config.checkpoint_every_n_epochs == 0 or is_best:
                    self._save_checkpoint(epoch + 1, model, optimizer, scheduler, is_best)
                
                self._save_state()
                
                # === Dual-metric early stopping ===
                if (self._state.epochs_no_upright_improve >= self.config.patience and 
                    self._state.epochs_no_rotation_improve >= self.config.patience):
                    logger.info(f"★ DUAL-METRIC early stopping at epoch {epoch+1}")
                    break
            
            # === STEP 8: Restore best model and save ===
            logger.info("Step 8: Finalizing...")
            if best_state is not None:
                model.load_state_dict(best_state)
                logger.info(f"Restored best model (upright: {self._state.best_upright_accuracy:.2f}%, rotation: {self._state.best_rotation_accuracy:.2f}%)")
            
            # Save final model
            self._save_final_model(model)
            
            # Archive feedback images (CRITICAL - prevents re-triggering)
            self._archive_feedback_images()
            
            # Update version in retrain_manager (MINOR increment)
            self._update_version_after_comprehensive()
            
            # Update state
            self._state.is_training = False
            self._state.is_completed = True
            self._state.completed_at = datetime.now().isoformat()
            self._save_state()
            
            logger.info("="*60)
            logger.info("🚀 COMPREHENSIVE TRAINING COMPLETED!")
            logger.info(f"  Best upright accuracy: {self._state.best_upright_accuracy:.2f}%")
            logger.info(f"  Best rotation accuracy: {self._state.best_rotation_accuracy:.2f}%")
            logger.info(f"  Total images used: {self._state.total_images_used}")
            logger.info("="*60)
            
        except Exception as e:
            logger.exception(f"Comprehensive training failed: {e}")
            self._state.is_training = False
            self._save_state()
    
    def _collect_data_paths(self) -> List[str]:
        """Collect all data source paths."""
        paths = []
        
        # Original datasets
        for dataset_path in self.config.dataset_paths:
            if Path(dataset_path).exists():
                paths.append(dataset_path)
                logger.info(f"  Added dataset: {dataset_path}")
        
        # Current feedback images
        if Path(self.config.feedback_images_dir).exists():
            paths.append(self.config.feedback_images_dir)
            logger.info(f"  Added feedback: {self.config.feedback_images_dir}")
        
        # Archived feedback images
        archive_dir = Path(self.config.archived_images_dir)
        if archive_dir.exists():
            for version_dir in archive_dir.iterdir():
                if version_dir.is_dir() and version_dir.name.startswith('v'):
                    paths.append(str(version_dir))
                    logger.info(f"  Added archive: {version_dir}")
        
        return paths
    
    def _create_balanced_dataloader(self, dataset: ComprehensiveDataset, device: str) -> DataLoader:
        """Create a balanced dataloader with oversampling."""
        if not self.config.oversample_minority:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=12,
                pin_memory=True if device == 'cuda' else False,
                persistent_workers=True
            )
        
        # Compute sample weights
        class_counts = [0] * len(dataset.class_names)
        for _, label, _ in dataset:
            class_counts[label] += 1
        
        # Reset and count properly
        class_counts = [0] * len(dataset.class_names)
        for _, label in dataset.samples:
            class_counts[label] += 1
        
        sample_weights = []
        for _, label in dataset.samples:
            class_count = class_counts[label]
            weight = 1.0 / class_count if class_count > 0 else 1.0
            sample_weights.append(weight)
        
        num_samples = max(class_counts) * len([c for c in class_counts if c > 0])
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=12,
            pin_memory=True if device == 'cuda' else False,
            persistent_workers=True
        )
    
    def _load_model(self, num_classes: int, device: str):
        """Load the model for training."""
        import sys
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Add KnowledgeDistillation src to path
        kd_src = Path("D:/KnowledgeDistillation/src")
        if kd_src.exists() and str(kd_src) not in sys.path:
            sys.path.insert(0, str(kd_src))
        
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        try:
            from enhanced_student_model import EnhancedStudentModel  # type: ignore
            
            stem_weight = state_dict.get('stem.0.weight')
            base_channels = stem_weight.shape[0] if stem_weight is not None else 48
            
            original_num_classes = 11
            model = EnhancedStudentModel(num_classes=original_num_classes, base_channels=base_channels)
            model.load_state_dict(state_dict, strict=False)
            
            # Expand classifier if needed
            if num_classes > original_num_classes:
                model = self._expand_classifier(model, num_classes)
            
            logger.info(f"Loaded EnhancedStudentModel with {num_classes} classes")
            
        except ImportError:
            logger.warning("EnhancedStudentModel not found, using MobileNetV3")
            import torchvision.models as models
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            model.load_state_dict(state_dict, strict=False)
        
        return model.to(device)
    
    def _expand_classifier(self, model, new_num_classes: int):
        """
        Expand classifier AND aux_classifiers for additional classes.
        
        CRITICAL: This preserves original class weights exactly and only initializes
        new class weights. The original 11 classes' learned patterns are untouched.
        """
        old_num_classes = None
        
        # 1. Expand main classifier (classifier.6 in Sequential)
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            for i in range(len(model.classifier) - 1, -1, -1):
                if isinstance(model.classifier[i], nn.Linear):
                    old_linear = model.classifier[i]
                    in_features = old_linear.in_features
                    old_num_classes = old_linear.out_features
                    
                    if old_num_classes >= new_num_classes:
                        logger.info(f"Main classifier already has {old_num_classes} classes")
                        break
                    
                    new_linear = nn.Linear(in_features, new_num_classes)
                    
                    # Copy old weights EXACTLY
                    with torch.no_grad():
                        new_linear.weight[:old_num_classes] = old_linear.weight.clone()
                        new_linear.bias[:old_num_classes] = old_linear.bias.clone()
                        # Small init for new class - prevents dominating softmax
                        nn.init.normal_(new_linear.weight[old_num_classes:], mean=0.0, std=0.01)
                        nn.init.constant_(new_linear.bias[old_num_classes:], -2.0)
                    
                    model.classifier[i] = new_linear
                    logger.info(f"Expanded main classifier: {old_num_classes} → {new_num_classes}")
                    break
        
        # 2. Expand aux_classifiers
        if hasattr(model, 'aux_classifiers') and old_num_classes is not None:
            aux_classifiers = model.aux_classifiers
            if isinstance(aux_classifiers, nn.ModuleDict):
                for stage_name, aux_clf in aux_classifiers.items():
                    if isinstance(aux_clf, nn.Sequential):
                        for j in range(len(aux_clf) - 1, -1, -1):
                            if isinstance(aux_clf[j], nn.Linear):
                                old_aux = aux_clf[j]
                                aux_in = old_aux.in_features
                                aux_old = old_aux.out_features
                                
                                if aux_old >= new_num_classes:
                                    break
                                
                                new_aux = nn.Linear(aux_in, new_num_classes)
                                with torch.no_grad():
                                    new_aux.weight[:aux_old] = old_aux.weight.clone()
                                    new_aux.bias[:aux_old] = old_aux.bias.clone()
                                    nn.init.normal_(new_aux.weight[aux_old:], mean=0.0, std=0.01)
                                    nn.init.constant_(new_aux.bias[aux_old:], -2.0)
                                
                                aux_clf[j] = new_aux
                                logger.info(f"Expanded aux_classifier[{stage_name}]: {aux_old} → {new_num_classes}")
                                break
        
        return model
    
    def _freeze_original_class_weights(self, model, num_original_classes: int = 11):
        """
        Freeze weights for original classes using gradient hooks.
        Only allows gradients for new class(es) to flow.
        """
        def make_gradient_hook(num_original: int, is_weight: bool = True):
            def hook(grad):
                if grad is None:
                    return grad
                new_grad = grad.clone()
                if is_weight:
                    new_grad[:num_original] = 0
                else:
                    new_grad[:num_original] = 0
                return new_grad
            return hook
        
        hooks = []
        
        # Hook main classifier
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            for layer in model.classifier:
                if isinstance(layer, nn.Linear) and layer.out_features > num_original_classes:
                    hook_w = layer.weight.register_hook(make_gradient_hook(num_original_classes, True))
                    hook_b = layer.bias.register_hook(make_gradient_hook(num_original_classes, False))
                    hooks.extend([hook_w, hook_b])
        
        # Hook aux_classifiers
        if hasattr(model, 'aux_classifiers'):
            for stage_name, aux_clf in model.aux_classifiers.items():
                if isinstance(aux_clf, nn.Sequential):
                    for layer in aux_clf:
                        if isinstance(layer, nn.Linear) and layer.out_features > num_original_classes:
                            hook_w = layer.weight.register_hook(make_gradient_hook(num_original_classes, True))
                            hook_b = layer.bias.register_hook(make_gradient_hook(num_original_classes, False))
                            hooks.extend([hook_w, hook_b])
        
        logger.info(f"Froze original {num_original_classes} classes with {len(hooks)} gradient hooks")
        return hooks
    
    def _backup_model(self):
        """Backup the original model before training."""
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(self.config.backup_dir) / f"comprehensive_backup_{timestamp}.pt"
        shutil.copy2(model_path, backup_path)
        logger.info(f"Model backed up to {backup_path}")
    
    def _setup_training(self, model, train_dataset, device):
        """Setup optimizer, scheduler, criterion, scaler, and KD components."""
        # Freeze backbone initially
        if self.config.freeze_backbone_epochs > 0:
            for name, param in model.named_parameters():
                if 'classifier' not in name and 'head' not in name:
                    param.requires_grad = False
            logger.info(f"Backbone frozen for first {self.config.freeze_backbone_epochs} epochs")
        
        # Class weights
        class_weights = None
        if self.config.use_class_weights:
            class_counts = train_dataset.get_class_counts()
            total = sum(class_counts.values())
            weights = []
            for name in train_dataset.class_names:
                count = class_counts.get(name, 1)
                weight = total / (len(class_counts) * count) if count > 0 else 1.0
                weights.append(weight)
            avg_weight = sum(weights) / len(weights)
            weights = [w / avg_weight for w in weights]
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
            logger.info(f"Class weights: {dict(zip(train_dataset.class_names, [f'{w:.2f}' for w in weights]))}")
        
        # Setup Knowledge Distillation (if enabled)
        sequential_kd = None
        kd_loss_fn = None
        if self.config.use_knowledge_distillation:
            try:
                from .kd_training import SequentialTeacherKD, KDConfig, KnowledgeDistillationLoss
                
                logger.info("Setting up Sequential Knowledge Distillation...")
                kd_config = KDConfig(
                    temperature=self.config.kd_temperature,
                    alpha=self.config.kd_alpha,
                    beta=self.config.kd_beta,
                    teacher_models_dir=self.config.teacher_models_dir,
                    use_student_as_teacher=self.config.use_student_as_teacher
                )
                
                # Load deployed student as a teacher (copy current model state)
                deployed_student_copy = None
                if self.config.use_student_as_teacher:
                    deployed_student_copy = copy.deepcopy(model)
                    deployed_student_copy.eval()
                
                num_classes = len(train_dataset.class_names)
                sequential_kd = SequentialTeacherKD(
                    config=kd_config,
                    deployed_student=deployed_student_copy,
                    num_classes=num_classes,
                    device=device,
                    epochs_per_teacher=self.config.epochs_per_teacher
                )
                
                kd_loss_fn = KnowledgeDistillationLoss(
                    temperature=self.config.kd_temperature,
                    alpha=self.config.kd_alpha,
                    beta=self.config.kd_beta,
                    class_weights=class_weights
                )
                
                teacher_count = sequential_kd.get_teacher_count()
                teacher_names = sequential_kd.get_teacher_names()
                logger.info(f"Sequential KD: {teacher_count} teachers, {self.config.epochs_per_teacher} epochs each")
                logger.info(f"Teachers: {teacher_names}")
                
                # Store for use in training
                self._sequential_kd = sequential_kd
                self._kd_loss_fn = kd_loss_fn
            except Exception as e:
                logger.warning(f"Could not initialize KD, falling back to CE loss: {e}")
                import traceback
                traceback.print_exc()
                self._sequential_kd = None
                self._kd_loss_fn = None
        
        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs - self.config.warmup_epochs,
            eta_min=self.config.min_learning_rate
        ) if self.config.use_cosine_schedule else None
        
        # Criterion (fallback CE loss)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Scaler
        scaler = GradScaler('cuda') if self.config.use_amp and device == 'cuda' else None
        
        return optimizer, scheduler, criterion, scaler
    
    def _unfreeze_backbone(self, model, optimizer):
        """Unfreeze backbone and recreate optimizer."""
        logger.info("Unfreezing backbone layers...")
        for param in model.parameters():
            param.requires_grad = True
        
        # Recreate optimizer with lower LR for backbone
        return optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate * 0.1,
            weight_decay=self.config.weight_decay
        )
    
    def _compute_fisher_information(self, model, dataloader, device):
        """Compute Fisher Information for EWC."""
        self._fisher_info = {}
        self._optimal_params = {}
        
        for name, param in model.named_parameters():
            self._optimal_params[name] = param.data.clone()
            self._fisher_info[name] = torch.zeros_like(param.data)
        
        model.eval()
        num_batches = min(100, len(dataloader))
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            images, labels, _ = batch
            images = images.to(device)
            labels = labels.to(device)
            
            model.zero_grad()
            outputs = model(images)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self._fisher_info[name] += param.grad.data ** 2
        
        for name in self._fisher_info:
            self._fisher_info[name] /= num_batches
        
        logger.info(f"Fisher Information computed from {num_batches} batches")
    
    def _ewc_loss(self, model) -> torch.Tensor:
        """Compute EWC penalty with clamping."""
        if self._fisher_info is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if name in self._fisher_info:
                ewc_loss += (self._fisher_info[name] * (param - self._optimal_params[name]) ** 2).sum()
        
        raw_loss = self.config.ewc_lambda * ewc_loss
        return torch.clamp(raw_loss, max=self.config.max_ewc_loss)
    
    def _train_epoch_with_teacher(self, model, dataloader, optimizer, criterion, scaler, device, epoch, teacher, kd_loss_fn):
        """Train one epoch with a SINGLE teacher for sequential KD."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, rotation_angles in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    outputs = model(images)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Get soft labels from SINGLE teacher
                    teacher_soft_labels = teacher.get_soft_labels(images)
                    base_loss, loss_dict = kd_loss_fn(logits, labels, teacher_soft_labels)
                    ce_loss = torch.tensor(loss_dict['ce_loss'], device=device)
                    
                    if self.config.use_ewc:
                        ewc_penalty = self._ewc_loss(model)
                        loss = base_loss + ewc_penalty
                    else:
                        loss = base_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                teacher_soft_labels = teacher.get_soft_labels(images)
                base_loss, loss_dict = kd_loss_fn(logits, labels, teacher_soft_labels)
                ce_loss = torch.tensor(loss_dict['ce_loss'], device=device)
                
                if self.config.use_ewc:
                    ewc_penalty = self._ewc_loss(model)
                    loss = base_loss + ewc_penalty
                else:
                    loss = base_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
            
            total_loss += ce_loss.item() if hasattr(ce_loss, 'item') else ce_loss
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def _train_epoch(self, model, dataloader, optimizer, criterion, scaler, device, epoch):
        """Train one epoch - Fallback CE loss only (when sequential KD not available)."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, rotation_angles in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    outputs = model(images)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    
                    # Fallback - CE loss only
                    base_loss = criterion(logits, labels)
                    ce_loss = base_loss
                    
                    if self.config.use_ewc:
                        ewc_penalty = self._ewc_loss(model)
                        loss = base_loss + ewc_penalty
                    else:
                        loss = base_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Fallback - CE loss only
                base_loss = criterion(logits, labels)
                ce_loss = base_loss
                
                if self.config.use_ewc:
                    ewc_penalty = self._ewc_loss(model)
                    loss = base_loss + ewc_penalty
                else:
                    loss = base_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
            
            total_loss += ce_loss.item() if hasattr(ce_loss, 'item') else ce_loss
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def _validate_upright(self, model, dataloader, criterion, device):
        """Validate on upright images."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images, labels, _ = batch
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def _validate_rotation(self, model, dataloader, device):
        """Validate on all rotation angles."""
        model.eval()
        
        # Test cardinal and some minor angles
        test_angles = [90, 180, 270, 45, 135, 225, 315]
        accuracies = []
        
        with torch.no_grad():
            for angle in test_angles:
                correct = 0
                total = 0
                
                for batch in dataloader:
                    images, labels, _ = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Apply rotation
                    if angle == 90:
                        rotated = torch.rot90(images, k=1, dims=[2, 3])
                    elif angle == 180:
                        rotated = torch.rot90(images, k=2, dims=[2, 3])
                    elif angle == 270:
                        rotated = torch.rot90(images, k=3, dims=[2, 3])
                    else:
                        # For non-90° angles, use grid_sample
                        theta = torch.tensor([
                            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0]
                        ], dtype=images.dtype, device=device).unsqueeze(0).expand(images.size(0), -1, -1)
                        grid = F.affine_grid(theta, images.size(), align_corners=False)
                        rotated = F.grid_sample(images, grid, align_corners=False)
                    
                    outputs = model(rotated)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                accuracies.append(100. * correct / total if total > 0 else 0)
        
        return sum(accuracies) / len(accuracies)
    
    def _get_current_version(self) -> tuple:
        """Get current version from retrain_manager."""
        try:
            from .retrain_manager import get_retrain_manager
            manager = get_retrain_manager()
            status = manager.get_status()
            return (
                status.version_major,
                status.version_minor,
                status.version_patch
            )
        except Exception as e:
            logger.warning(f"Could not get current version: {e}")
            return (1, 0, 0)
    
    def _update_version_after_comprehensive(self):
        """
        Update version in retrain_manager after comprehensive training.
        Increments MINOR version and resets PATCH to 0.
        """
        try:
            from .retrain_manager import get_retrain_manager
            manager = get_retrain_manager()
            
            # Get current and calculate new version
            with manager._lock:
                old_version = manager._status.current_version_string
                manager._status.version_minor += 1  # Increment MINOR
                manager._status.version_patch = 0    # Reset PATCH
                manager._status.total_comprehensive += 1
                new_version = manager._status.current_version_string
            
            # Save the updated status
            manager._save_status()
            
            # Copy model with new version name
            model_path = Path(self.config.model_path)
            versioned_name = f"student_model_{new_version}.pt"
            versioned_path = model_path.parent / versioned_name
            
            if model_path.exists():
                shutil.copy2(model_path, versioned_path)
                logger.info(f"📦 Versioned model saved: {versioned_path}")
            
            logger.info(f"🚀 Version updated: {old_version} → {new_version} (comprehensive training)")
            
        except Exception as e:
            logger.error(f"Could not update version after comprehensive training: {e}")
    
    def _save_final_model(self, model):
        """Save the final trained model."""
        model_path = Path(self.config.model_path)
        
        # Get version info
        major, minor, patch = self._get_current_version()
        new_version = f"v{major}.{minor + 1}.0"  # Comprehensive increments MINOR, resets PATCH
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'class_names': self.config.class_names + (['junk'] if self.config.include_junk_class else []),
            'comprehensive_trained': True,
            'training_date': datetime.now().isoformat(),
            'best_upright_accuracy': self._state.best_upright_accuracy,
            'best_rotation_accuracy': self._state.best_rotation_accuracy,
            'total_images_used': self._state.total_images_used,
            'training_version': self._state.training_version,
            # Semantic version
            'model_version': new_version,
            'version_major': major,
            'version_minor': minor + 1,
            'version_patch': 0,
            'training_type': 'comprehensive',
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"Final model saved to {model_path} (version {new_version})")

    def _archive_feedback_images(self):
        """
        Move trained feedback images to archive folder.
        This prevents the same images from triggering immediate re-training.
        Images are moved to: {history_dir}/v{version}_comprehensive/
        """
        version = self._state.training_version or "1"
        archive_dir = Path(self.config.archived_images_dir) / f"v{version}_comprehensive"
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
        
        logger.info(f"📦 Archived {archived_count} feedback images to {archive_dir}")


# ============================================================
# Global Instance and Trigger Check
# ============================================================

_comprehensive_trainer: Optional[ComprehensiveTrainer] = None


def get_comprehensive_trainer() -> ComprehensiveTrainer:
    """Get or create the global comprehensive trainer."""
    global _comprehensive_trainer
    if _comprehensive_trainer is None:
        _comprehensive_trainer = ComprehensiveTrainer()
    return _comprehensive_trainer


def check_comprehensive_training_trigger(total_feedback_count: int) -> bool:
    """
    Check if comprehensive training should be triggered.
    
    Args:
        total_feedback_count: Total number of feedback images across all history
        
    Returns:
        True if training was triggered
    """
    trainer = get_comprehensive_trainer()
    
    if trainer.is_training():
        return False
    
    if trainer.get_state().is_completed:
        return False
    
    if total_feedback_count >= trainer.config.trigger_threshold:
        logger.info(f"🚀 Comprehensive training triggered! Total feedback: {total_feedback_count}")
        return trainer.start_training(resume=True)
    
    return False


def get_comprehensive_training_status() -> Dict[str, Any]:
    """Get comprehensive training status."""
    trainer = get_comprehensive_trainer()
    state = trainer.get_state()
    
    return {
        "is_training": state.is_training,
        "is_completed": state.is_completed,
        "can_resume": trainer.can_resume(),
        "current_epoch": state.current_epoch,
        "total_epochs": state.total_epochs,
        "best_upright_accuracy": state.best_upright_accuracy,
        "best_rotation_accuracy": state.best_rotation_accuracy,
        "current_upright_accuracy": state.current_upright_accuracy,
        "current_rotation_accuracy": state.current_rotation_accuracy,
        "total_images_used": state.total_images_used,
        "started_at": state.started_at,
        "completed_at": state.completed_at,
        "last_checkpoint_at": state.last_checkpoint_at,
    }


if __name__ == "__main__":
    # For manual testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Training")
    parser.add_argument("--start", action="store_true", help="Start training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--status", action="store_true", help="Show status")
    
    args = parser.parse_args()
    
    trainer = get_comprehensive_trainer()
    
    if args.status:
        status = get_comprehensive_training_status()
        print(json.dumps(status, indent=2))
    elif args.start:
        trainer.start_training(resume=args.resume)
        # Wait for training to complete
        while trainer.is_training():
            time.sleep(10)
            state = trainer.get_state()
            print(f"Epoch {state.current_epoch}/{state.total_epochs} - "
                  f"Upright: {state.current_upright_accuracy:.2f}% - "
                  f"Rotation: {state.current_rotation_accuracy:.2f}%")
    else:
        parser.print_help()
