"""
Training module for model retraining.
"""

from .comprehensive_trainer import (
    ComprehensiveTrainer,
    ComprehensiveTrainingConfig,
    ComprehensiveTrainingState,
    check_comprehensive_training_trigger,
    get_comprehensive_trainer,
    get_comprehensive_training_status,
)
from .kd_training import (
    KDConfig,
    KnowledgeDistillationLoss,
    ONNXTeacher,
    PyTorchTeacher,
    TeacherEnsemble,
    create_teacher_ensemble,
    get_teacher_soft_labels,
)
from .retrain_manager import (
    FeedbackImageDataset,
    ModelRetrainingManager,
    RetrainingConfig,
    RetrainingStatus,
    check_retrain_status,
    get_retrain_manager,
)
from .version_manager import (
    ModelVersion,
    ModelVersionManager,
    VersionRecord,
    get_version_manager,
    init_version_manager,
)

__all__ = [
    # Regular retraining
    'ModelRetrainingManager',
    'RetrainingConfig',
    'RetrainingStatus',
    'FeedbackImageDataset',
    'get_retrain_manager',
    'check_retrain_status',
    # Comprehensive training
    'ComprehensiveTrainer',
    'ComprehensiveTrainingConfig',
    'ComprehensiveTrainingState',
    'get_comprehensive_trainer',
    'check_comprehensive_training_trigger',
    'get_comprehensive_training_status',
    # Version management
    'ModelVersion',
    'ModelVersionManager',
    'VersionRecord',
    'get_version_manager',
    'init_version_manager',
    # Knowledge Distillation
    'TeacherEnsemble',
    'KDConfig',
    'KnowledgeDistillationLoss',
    'ONNXTeacher',
    'PyTorchTeacher',
    'create_teacher_ensemble',
    'get_teacher_soft_labels',
]
