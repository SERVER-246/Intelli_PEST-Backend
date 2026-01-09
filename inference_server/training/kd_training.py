"""
Knowledge Distillation Training Module
=====================================
Provides multi-teacher knowledge distillation for model retraining.

Uses all 11 teacher models (ONNX format) + the currently deployed student
model to generate soft labels for training new feedback images.

Teacher Models:
1. mobilenet_v2
2. resnet50  
3. inception_v3
4. efficientnet_b0
5. darknet53
6. alexnet
7. yolo11n-cls
8. ensemble_attention
9. ensemble_concat
10. ensemble_cross
11. super_ensemble

The deployed student model is also used as a teacher, giving 12 total teachers.

Author: IntelliPEST KD Pipeline
Date: 2026-01-08
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Optional imports
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not available - ONNX teachers will be skipped")


@dataclass
class TeacherConfig:
    """Configuration for a single teacher model."""
    name: str
    path: str
    weight: float = 1.0
    enabled: bool = True


# List of ensemble models that need TorchScript (their ONNX exports have output scale issues)
TORCHSCRIPT_ENSEMBLE_MODELS = [
    'ensemble_cross',
    'super_ensemble',
    'ensemble_attention',
    'ensemble_concat',
]


@dataclass 
class KDConfig:
    """Knowledge Distillation configuration."""
    # Temperature for soft labels (higher = softer)
    temperature: float = 4.0
    
    # Loss weights: alpha * hard_label_loss + beta * soft_label_loss
    alpha: float = 0.3  # Hard label (ground truth) weight
    beta: float = 0.7   # Soft label (teacher) weight
    
    # Teacher model settings
    teacher_models_dir: str = "D:/Intelli_PEST-Backend/tflite_models_compatible/onnx_models"
    # Directory containing TorchScript (.pt) ensemble models (local backup from G: drive)
    torchscript_models_dir: str = "D:/Intelli_PEST-Backend/teacher_models/torchscript"
    use_student_as_teacher: bool = True  # Include deployed student as a teacher
    
    # Default teacher weights (higher = more influence)
    teacher_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.teacher_weights is None:
            # Default weights - ensembles get higher weight
            self.teacher_weights = {
                'mobilenet_v2': 1.0,
                'resnet50': 1.0,
                'inception_v3': 1.0,
                'efficientnet_b0': 1.0,
                'darknet53': 1.0,
                'alexnet': 1.0,
                'yolo11n-cls': 1.0,
                'ensemble_attention': 1.5,
                'ensemble_concat': 1.5,
                'ensemble_cross': 1.5,
                'super_ensemble': 2.0,
                'deployed_student': 1.5,  # Current deployed model
            }


class ONNXTeacher:
    """Single ONNX teacher model wrapper."""
    
    def __init__(self, name: str, path: Path, weight: float = 1.0, num_classes: int = 11):
        self.name = name
        self.path = path
        self.weight = weight
        self.num_classes = num_classes
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load()
    
    def _load(self):
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            logger.warning(f"Skipping ONNX {self.name}: onnxruntime not available")
            return
        
        try:
            # Use CPU for reliability
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(self.path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"  Loaded ONNX teacher: {self.name} (weight={self.weight})")
        except Exception as e:
            logger.error(f"  Failed to load ONNX {self.name}: {e}")
            self.session = None
    
    def predict(self, images: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference on images.
        
        Args:
            images: Input as numpy array (B, C, H, W) in float32, normalized
            
        Returns:
            Logits as numpy array (B, num_classes) or None if failed
        """
        if self.session is None:
            return None
        
        try:
            outputs = self.session.run(None, {self.input_name: images.astype(np.float32)})
            return outputs[0]
        except Exception as e:
            logger.error(f"ONNX inference error for {self.name}: {e}")
            return None
    
    def is_loaded(self) -> bool:
        return self.session is not None


class TorchScriptTeacher:
    """
    TorchScript teacher model wrapper.
    Used for ensemble models where ONNX export has output scale issues.
    Loads JIT-compiled .pt files from deployment directory.
    """
    
    def __init__(self, name: str, path: Path, weight: float = 1.0, device: str = 'cpu'):
        self.name = name
        self.path = path
        self.weight = weight
        self.device = device
        self.model = None
        self.num_classes = 11  # Default
        self._load()
    
    def _load(self):
        """Load TorchScript model."""
        try:
            self.model = torch.jit.load(str(self.path), map_location=self.device)
            self.model.eval()
            
            # Detect number of output classes
            with torch.no_grad():
                dummy = torch.randn(1, 3, 256, 256, device=self.device)
                output = self.model(dummy)
                self.num_classes = output.shape[1]
            
            logger.info(f"  Loaded TorchScript teacher: {self.name} (weight={self.weight}, classes={self.num_classes})")
        except Exception as e:
            logger.error(f"  Failed to load TorchScript {self.name}: {e}")
            self.model = None
    
    def predict(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Run inference on images.
        
        Args:
            images: Input as torch tensor (B, C, H, W)
            
        Returns:
            Logits as torch tensor (B, num_classes) or None if failed
        """
        if self.model is None:
            return None
        
        try:
            with torch.no_grad():
                images = images.to(self.device)
                outputs = self.model(images)
                return outputs
        except Exception as e:
            logger.error(f"TorchScript inference error for {self.name}: {e}")
            return None
    
    def is_loaded(self) -> bool:
        return self.model is not None


class PyTorchTeacher:
    """PyTorch teacher model wrapper (for deployed student)."""
    
    def __init__(self, name: str, model: nn.Module, weight: float = 1.0, device: str = 'cuda'):
        self.name = name
        self.model = model
        self.weight = weight
        self.device = device
        self.num_classes = self._detect_num_classes(model)
        
        if model is not None:
            self.model.eval()
            logger.info(f"  Loaded PyTorch teacher: {self.name} (weight={self.weight}, classes={self.num_classes})")
    
    def _detect_num_classes(self, model: nn.Module) -> int:
        """Detect number of output classes from the model."""
        if model is None:
            return 11  # Default fallback
        
        try:
            # Try to find the final Linear layer
            if hasattr(model, 'classifier'):
                classifier = model.classifier
                if isinstance(classifier, nn.Sequential):
                    for layer in reversed(list(classifier.children())):
                        if isinstance(layer, nn.Linear):
                            return layer.out_features
                elif isinstance(classifier, nn.Linear):
                    return classifier.out_features
            
            # Try direct fc layer
            if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                return model.fc.out_features
            
            # Fallback: run inference to detect
            with torch.no_grad():
                dummy = torch.randn(1, 3, 256, 256).to(self.device)
                output = model(dummy)
                if isinstance(output, dict):
                    output = output.get('logits', list(output.values())[0])
                return output.shape[1]
        except Exception:
            return 11  # Default fallback
    
    def predict(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Run inference on images.
        
        Args:
            images: Input as torch tensor (B, C, H, W)
            
        Returns:
            Logits as torch tensor (B, num_classes)
        """
        if self.model is None:
            return None
        
        try:
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    return outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
                return outputs
        except Exception as e:
            logger.error(f"PyTorch inference error for {self.name}: {e}")
            return None
    
    def is_loaded(self) -> bool:
        return self.model is not None


class TeacherEnsemble:
    """
    Ensemble of all teacher models for knowledge distillation.
    
    Loads 11 ONNX teacher models + optionally the deployed PyTorch student.
    Generates weighted soft labels from all teachers.
    """
    
    # List of all teacher model names
    TEACHER_MODELS = [
        'mobilenet_v2',
        'resnet50',
        'inception_v3',
        'efficientnet_b0',
        'darknet53',
        'alexnet',
        'yolo11n-cls',
        'ensemble_attention',
        'ensemble_concat',
        'ensemble_cross',
        'super_ensemble',
    ]
    
    def __init__(
        self,
        config: KDConfig,
        deployed_student: Optional[nn.Module] = None,
        num_classes: int = 11,
        device: str = 'cuda'
    ):
        """
        Initialize teacher ensemble.
        
        Args:
            config: KD configuration
            deployed_student: Currently deployed student model (optional)
            num_classes: Number of output classes
            device: Device for PyTorch operations
        """
        self.config = config
        self.num_classes = num_classes
        self.device = device
        self.onnx_teachers: Dict[str, ONNXTeacher] = {}
        self.pytorch_teacher: Optional[PyTorchTeacher] = None
        self.total_weight = 0.0
        
        # Load all teachers
        self._load_onnx_teachers()
        
        # Add deployed student as teacher if provided
        if config.use_student_as_teacher and deployed_student is not None:
            self._add_student_teacher(deployed_student)
        
        # Calculate teacher class counts for logging
        onnx_classes = 11  # ONNX teachers always have 11 classes
        pytorch_classes = self.pytorch_teacher.num_classes if self.pytorch_teacher else None
        
        logger.info(f"Teacher ensemble ready: {len(self.onnx_teachers)} ONNX + "
                   f"{1 if self.pytorch_teacher else 0} PyTorch teachers")
        logger.info(f"Total teacher weight: {self.total_weight:.2f}")
        logger.info(f"Student output classes: {self.num_classes}")
        logger.info(f"ONNX teacher classes: {onnx_classes}, PyTorch teacher classes: {pytorch_classes}")
        if self.num_classes != onnx_classes:
            logger.info(f"  -> Dimension mismatch will be handled by padding (teachers {onnx_classes} -> student {self.num_classes})")
    
    def _load_onnx_teachers(self):
        """Load all ONNX teacher models."""
        models_dir = Path(self.config.teacher_models_dir)
        
        if not models_dir.exists():
            logger.error(f"Teacher models directory not found: {models_dir}")
            return
        
        logger.info(f"Loading teacher models from: {models_dir}")
        
        for name in self.TEACHER_MODELS:
            path = models_dir / f"{name}.onnx"
            weight = self.config.teacher_weights.get(name, 1.0)
            
            if not path.exists():
                logger.warning(f"  Teacher not found: {path}")
                continue
            
            teacher = ONNXTeacher(name, path, weight, self.num_classes)
            if teacher.is_loaded():
                self.onnx_teachers[name] = teacher
                self.total_weight += weight
    
    def _add_student_teacher(self, model: nn.Module):
        """Add the deployed student model as an additional teacher."""
        weight = self.config.teacher_weights.get('deployed_student', 1.5)
        self.pytorch_teacher = PyTorchTeacher(
            name='deployed_student',
            model=model,
            weight=weight,
            device=self.device
        )
        if self.pytorch_teacher.is_loaded():
            self.total_weight += weight
    
    def get_soft_labels(
        self,
        images: torch.Tensor,
        temperature: float = None
    ) -> torch.Tensor:
        """
        Generate weighted soft labels from all teachers.
        
        Args:
            images: Input images (B, C, H, W) as torch tensor
            temperature: Temperature for softmax (default from config)
            
        Returns:
            Soft labels as torch tensor (B, num_classes)
        """
        if temperature is None:
            temperature = self.config.temperature
        
        batch_size = images.shape[0]
        device = images.device
        
        # Accumulate weighted soft labels
        weighted_sum = torch.zeros(batch_size, self.num_classes, device=device)
        
        # Convert to numpy for ONNX inference
        images_np = images.cpu().numpy()
        
        # Helper function for parallel ONNX teacher inference
        def process_onnx_teacher(name_teacher_tuple):
            name, teacher = name_teacher_tuple
            logits = teacher.predict(images_np)
            if logits is not None:
                return (name, logits, teacher.weight)
            return None
        
        # Get predictions from all ONNX teachers IN PARALLEL
        with ThreadPoolExecutor(max_workers=min(8, len(self.onnx_teachers))) as executor:
            futures = {executor.submit(process_onnx_teacher, item): item[0] 
                      for item in self.onnx_teachers.items()}
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    name, logits, weight = result
                    # Convert to tensor and apply temperature-scaled softmax
                    logits_tensor = torch.from_numpy(logits).float().to(device)
                    teacher_classes = logits_tensor.shape[1]
                    
                    # Handle class count mismatch (ONNX teachers have 11, student may have 12 with junk)
                    if teacher_classes < self.num_classes:
                        # Pad with low logit values for extra classes (junk)
                        # -10.0 gives very low probability after softmax (~0.00005)
                        # This means "teachers don't know about junk class"
                        padding = torch.full(
                            (batch_size, self.num_classes - teacher_classes),
                            fill_value=-10.0,
                            device=device
                        )
                        logits_tensor = torch.cat([logits_tensor, padding], dim=1)
                    elif teacher_classes > self.num_classes:
                        # Truncate if teacher has more classes (shouldn't happen)
                        logger.warning(f"Teacher {name} has {teacher_classes} classes > student {self.num_classes}, truncating")
                        logits_tensor = logits_tensor[:, :self.num_classes]
                    
                    soft_probs = F.softmax(logits_tensor / temperature, dim=1)
                    weighted_sum += soft_probs * weight
        
        # Get prediction from PyTorch teacher (deployed student)
        if self.pytorch_teacher is not None and self.pytorch_teacher.is_loaded():
            logits = self.pytorch_teacher.predict(images)
            if logits is not None:
                teacher_classes = logits.shape[1]
                
                # Handle class count mismatch
                # Note: deployed student may have been expanded to 12 classes before being copied
                if teacher_classes < self.num_classes:
                    # Pad if PyTorch teacher has fewer classes
                    padding = torch.full(
                        (batch_size, self.num_classes - teacher_classes),
                        fill_value=-10.0,
                        device=device
                    )
                    logits = torch.cat([logits, padding], dim=1)
                elif teacher_classes > self.num_classes:
                    # Truncate if teacher has more classes
                    logger.warning(f"PyTorch teacher has {teacher_classes} classes > student {self.num_classes}, truncating")
                    logits = logits[:, :self.num_classes]
                # If teacher_classes == self.num_classes, no adjustment needed
                
                soft_probs = F.softmax(logits / temperature, dim=1)
                weighted_sum += soft_probs * self.pytorch_teacher.weight
        
        # Normalize by total weight
        if self.total_weight > 0:
            soft_labels = weighted_sum / self.total_weight
        else:
            # Fallback to uniform if no teachers loaded
            soft_labels = torch.ones(batch_size, self.num_classes, device=device) / self.num_classes
        
        return soft_labels
    
    def get_teacher_count(self) -> int:
        """Return total number of loaded teachers."""
        count = len(self.onnx_teachers)
        if self.pytorch_teacher is not None and self.pytorch_teacher.is_loaded():
            count += 1
        return count
    
    def get_teacher_names(self) -> List[str]:
        """Return names of all loaded teachers."""
        names = list(self.onnx_teachers.keys())
        if self.pytorch_teacher is not None and self.pytorch_teacher.is_loaded():
            names.append('deployed_student')
        return names


class KnowledgeDistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation training.
    
    Loss = alpha * CE(student_output, hard_labels) + beta * KL(student_soft, teacher_soft)
    
    Where:
    - hard_labels: Ground truth from feedback
    - teacher_soft: Soft labels from teacher ensemble
    - student_soft: Student's temperature-scaled predictions
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize KD loss.
        
        Args:
            temperature: Temperature for soft labels
            alpha: Weight for hard label (CE) loss
            beta: Weight for soft label (KL) loss
            class_weights: Optional class weights for CE loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        teacher_soft_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined KD loss.
        
        Args:
            student_logits: Student model output (B, num_classes)
            hard_labels: Ground truth labels (B,)
            teacher_soft_labels: Soft labels from teachers (B, num_classes)
            
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary with individual loss components
        """
        student_classes = student_logits.shape[1]
        teacher_classes = teacher_soft_labels.shape[1]
        
        # SAFETY CHECK: Ensure dimensions match
        if student_classes != teacher_classes:
            # This shouldn't happen if TeacherEnsemble.get_soft_labels() works correctly
            # But handle it gracefully just in case
            if teacher_classes < student_classes:
                # Pad teacher soft labels with small probability for extra classes
                padding = torch.full(
                    (teacher_soft_labels.shape[0], student_classes - teacher_classes),
                    fill_value=1e-8,
                    device=teacher_soft_labels.device
                )
                teacher_soft_labels = torch.cat([teacher_soft_labels, padding], dim=1)
                teacher_soft_labels = teacher_soft_labels / teacher_soft_labels.sum(dim=1, keepdim=True)
            else:
                # Truncate teacher soft labels (shouldn't happen)
                teacher_soft_labels = teacher_soft_labels[:, :student_classes]
                teacher_soft_labels = teacher_soft_labels / teacher_soft_labels.sum(dim=1, keepdim=True)
            logger.warning(f"KD Loss dimension mismatch fixed: teacher {teacher_classes} -> student {student_classes}")
        
        # Hard label loss (standard cross-entropy)
        ce_loss = self.ce_loss(student_logits, hard_labels)
        
        # Soft label loss (KL divergence)
        # Temperature-scaled student softmax
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Ensure teacher soft labels are valid probabilities (no zeros for log stability)
        teacher_soft_labels = teacher_soft_labels.clamp(min=1e-8)
        teacher_soft_labels = teacher_soft_labels / teacher_soft_labels.sum(dim=1, keepdim=True)
        
        # KL divergence (teacher_soft is already probabilities)
        # KL(teacher || student) = sum(teacher * log(teacher/student))
        # Using F.kl_div which expects log_probs for input
        kl_loss = F.kl_div(
            student_soft,
            teacher_soft_labels,
            reduction='batchmean'
        ) * (self.temperature ** 2)  # Scale by T^2 as in Hinton et al.
        
        # Combined loss
        total_loss = self.alpha * ce_loss + self.beta * kl_loss
        
        loss_dict = {
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, loss_dict


def create_teacher_ensemble(
    deployed_student_model: Optional[nn.Module] = None,
    models_dir: str = "D:/Intelli_PEST-Backend/tflite_models_compatible/onnx_models",
    num_classes: int = 12,  # 11 pests + junk
    device: str = 'cuda',
    temperature: float = 4.0,
    alpha: float = 0.3,
    beta: float = 0.7,
) -> Tuple[TeacherEnsemble, KnowledgeDistillationLoss]:
    """
    Factory function to create teacher ensemble and KD loss.
    
    Args:
        deployed_student_model: Currently deployed student (for using as teacher)
        models_dir: Directory containing ONNX teacher models
        num_classes: Number of output classes
        device: Device for PyTorch operations
        temperature: Temperature for soft labels
        alpha: Hard label loss weight
        beta: Soft label loss weight
        
    Returns:
        teacher_ensemble: TeacherEnsemble instance
        kd_loss: KnowledgeDistillationLoss instance
    """
    config = KDConfig(
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        teacher_models_dir=models_dir,
        use_student_as_teacher=(deployed_student_model is not None)
    )
    
    ensemble = TeacherEnsemble(
        config=config,
        deployed_student=deployed_student_model,
        num_classes=num_classes,
        device=device
    )
    
    kd_loss = KnowledgeDistillationLoss(
        temperature=temperature,
        alpha=alpha,
        beta=beta
    )
    
    return ensemble, kd_loss


# Convenience function for getting soft labels in training loop
def get_teacher_soft_labels(
    ensemble: TeacherEnsemble,
    images: torch.Tensor,
    temperature: float = 4.0
) -> torch.Tensor:
    """
    Get soft labels from teacher ensemble for a batch of images.
    
    Args:
        ensemble: TeacherEnsemble instance
        images: Input images (B, C, H, W)
        temperature: Temperature for softmax
        
    Returns:
        Soft labels (B, num_classes)
    """
    return ensemble.get_soft_labels(images, temperature)


class SingleTeacher:
    """
    Single teacher wrapper for sequential training.
    Loads ONE teacher at a time to minimize memory usage.
    Supports ONNX, TorchScript, and PyTorch models.
    """
    
    def __init__(
        self,
        teacher_name: str,
        teacher_path: Path,
        weight: float,
        num_classes: int,
        temperature: float,
        device: str = 'cuda',
        is_pytorch: bool = False,
        is_torchscript: bool = False,
        pytorch_model: Optional[nn.Module] = None
    ):
        self.name = teacher_name
        self.path = teacher_path
        self.weight = weight
        self.num_classes = num_classes
        self.temperature = temperature
        self.device = device
        self.is_pytorch = is_pytorch
        self.is_torchscript = is_torchscript
        self.teacher = None
        self.teacher_num_classes = 11  # Default for ONNX teachers
        
        if is_pytorch and pytorch_model is not None:
            self.teacher = PyTorchTeacher(
                name=teacher_name,
                model=pytorch_model,
                weight=weight,
                device=device
            )
            self.teacher_num_classes = self.teacher.num_classes
        elif is_torchscript and teacher_path and teacher_path.exists():
            # Load TorchScript model for ensemble teachers
            self.teacher = TorchScriptTeacher(
                name=teacher_name,
                path=teacher_path,
                weight=weight,
                device='cpu'  # TorchScript ensembles are large, use CPU
            )
            self.teacher_num_classes = self.teacher.num_classes
        elif not is_pytorch and not is_torchscript and teacher_path and teacher_path.exists():
            self.teacher = ONNXTeacher(teacher_name, teacher_path, weight, num_classes)
        
        if self.teacher and self.teacher.is_loaded():
            model_type = "TorchScript" if is_torchscript else ("PyTorch" if is_pytorch else "ONNX")
            logger.info(f"  ✓ Loaded {model_type} teacher: {teacher_name} (weight={weight}, classes={self.teacher_num_classes})")
        else:
            logger.warning(f"  ✗ Failed to load teacher: {teacher_name}")
    
    def get_soft_labels(self, images: torch.Tensor) -> torch.Tensor:
        """Get soft labels from this single teacher."""
        if self.teacher is None or not self.teacher.is_loaded():
            # Return uniform distribution
            batch_size = images.size(0)
            return torch.ones(batch_size, self.num_classes, device=self.device) / self.num_classes
        
        batch_size = images.size(0)
        device = images.device
        
        if self.is_pytorch or self.is_torchscript:
            # PyTorch and TorchScript teachers use torch tensors
            logits = self.teacher.predict(images)
            if logits is not None:
                logits = logits.to(device)
        else:
            # ONNX teachers use numpy arrays
            images_np = images.cpu().numpy()
            logits_np = self.teacher.predict(images_np)
            if logits_np is None:
                return torch.ones(batch_size, self.num_classes, device=device) / self.num_classes
            logits = torch.from_numpy(logits_np).float().to(device)
        
        if logits is None:
            return torch.ones(batch_size, self.num_classes, device=device) / self.num_classes
        
        # Handle dimension mismatch (teacher 11 classes -> student 12 classes)
        teacher_classes = logits.shape[1]
        if teacher_classes < self.num_classes:
            padding = torch.full(
                (batch_size, self.num_classes - teacher_classes),
                fill_value=-10.0,
                device=device
            )
            logits = torch.cat([logits, padding], dim=1)
        elif teacher_classes > self.num_classes:
            logits = logits[:, :self.num_classes]
        
        # Apply temperature-scaled softmax
        soft_labels = F.softmax(logits / self.temperature, dim=1)
        return soft_labels
    
    def unload(self):
        """Unload teacher to free memory."""
        if self.teacher is not None:
            if hasattr(self.teacher, 'session'):
                self.teacher.session = None
            if hasattr(self.teacher, 'model'):
                self.teacher.model = None
            self.teacher = None
            logger.info(f"  Unloaded teacher: {self.name}")


class SequentialTeacherKD:
    """
    Sequential Knowledge Distillation - trains with ONE teacher at a time.
    
    Workflow:
    1. Load teacher 1 (e.g., alexnet)
    2. Train for N epochs with teacher 1
    3. Unload teacher 1
    4. Load teacher 2 (e.g., resnet50)
    5. Train for N epochs with teacher 2
    6. ... repeat for all teachers
    
    Benefits:
    - Low memory usage (only 1 teacher loaded at a time)
    - Each teacher gets focused training
    - More stable gradients
    """
    
    # All available teachers in training order
    TEACHER_ORDER = [
        'alexnet',           # Simplest first
        'mobilenet_v2',
        'efficientnet_b0',
        'yolo11n-cls',
        'resnet50',
        'darknet53',
        'inception_v3',
        'ensemble_attention',
        'ensemble_concat',
        'ensemble_cross',
        'super_ensemble',    # Most complex
        'deployed_student',  # Current model as teacher (last)
    ]
    
    # Teacher weights (same as before)
    TEACHER_WEIGHTS = {
        'mobilenet_v2': 1.0,
        'resnet50': 1.0,
        'inception_v3': 1.0,
        'efficientnet_b0': 1.0,
        'darknet53': 1.0,
        'alexnet': 1.0,
        'yolo11n-cls': 1.0,
        'ensemble_attention': 1.5,
        'ensemble_concat': 1.5,
        'ensemble_cross': 1.5,
        'super_ensemble': 2.0,
        'deployed_student': 1.5,
    }
    
    def __init__(
        self,
        config: KDConfig,
        deployed_student: Optional[nn.Module] = None,
        num_classes: int = 12,
        device: str = 'cuda',
        epochs_per_teacher: int = 2
    ):
        self.config = config
        self.num_classes = num_classes
        self.device = device
        self.deployed_student = deployed_student
        self.models_dir = Path(config.teacher_models_dir)
        self.torchscript_dir = Path(config.torchscript_models_dir)
        self.epochs_per_teacher = epochs_per_teacher
        
        # Verify available teachers
        self.available_teachers = []
        for name in self.TEACHER_ORDER:
            if name == 'deployed_student':
                if deployed_student is not None:
                    self.available_teachers.append(name)
            elif name in TORCHSCRIPT_ENSEMBLE_MODELS:
                # Check for TorchScript model in local backup directory
                ts_path = self.torchscript_dir / f"{name}.pt"
                if ts_path.exists():
                    self.available_teachers.append(name)
                    logger.info(f"  Found TorchScript ensemble: {name}")
                else:
                    logger.warning(f"  TorchScript not found for {name}: {ts_path}")
            else:
                path = self.models_dir / f"{name}.onnx"
                if path.exists():
                    self.available_teachers.append(name)
        
        logger.info(f"Sequential KD initialized with {len(self.available_teachers)} teachers")
        logger.info(f"Teacher order: {self.available_teachers}")
        logger.info(f"ONNX models dir: {self.models_dir}")
        logger.info(f"TorchScript models dir: {self.torchscript_dir}")
        logger.info(f"Student output classes: {num_classes}")
    
    def get_teacher_count(self) -> int:
        return len(self.available_teachers)
    
    def get_teacher_names(self) -> List[str]:
        return self.available_teachers.copy()
    
    def load_teacher(self, teacher_name: str) -> SingleTeacher:
        """Load a single teacher by name."""
        weight = self.TEACHER_WEIGHTS.get(teacher_name, 1.0)
        
        if teacher_name == 'deployed_student':
            return SingleTeacher(
                teacher_name=teacher_name,
                teacher_path=None,
                weight=weight,
                num_classes=self.num_classes,
                temperature=self.config.temperature,
                device=self.device,
                is_pytorch=True,
                pytorch_model=self.deployed_student
            )
        elif teacher_name in TORCHSCRIPT_ENSEMBLE_MODELS:
            # Load TorchScript model for ensemble teachers (from local backup)
            ts_path = self.torchscript_dir / f"{teacher_name}.pt"
            return SingleTeacher(
                teacher_name=teacher_name,
                teacher_path=ts_path,
                weight=weight,
                num_classes=self.num_classes,
                temperature=self.config.temperature,
                device=self.device,
                is_pytorch=False,
                is_torchscript=True
            )
        else:
            # Load ONNX model for single architecture teachers
            path = self.models_dir / f"{teacher_name}.onnx"
            return SingleTeacher(
                teacher_name=teacher_name,
                teacher_path=path,
                weight=weight,
                num_classes=self.num_classes,
                temperature=self.config.temperature,
                device=self.device,
                is_pytorch=False,
                is_torchscript=False
            )
    
    def iterate_teachers(self):
        """
        Generator that yields teachers one at a time.
        Automatically loads and unloads each teacher.
        
        Usage:
            for teacher_name, teacher, phase_num, total_phases in seq_kd.iterate_teachers():
                # Train with this teacher
                for epoch in range(epochs_per_teacher):
                    ...
                # Teacher is automatically unloaded when loop continues
        """
        total = len(self.available_teachers)
        for idx, name in enumerate(self.available_teachers):
            logger.info(f"\n{'='*60}")
            logger.info(f"PHASE {idx+1}/{total}: Training with teacher '{name}'")
            logger.info(f"{'='*60}")
            
            teacher = self.load_teacher(name)
            yield name, teacher, idx + 1, total
            teacher.unload()
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
