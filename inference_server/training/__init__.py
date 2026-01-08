"""
Training module for model retraining.
"""

from .retrain_manager import (
    ModelRetrainingManager,
    RetrainingConfig,
    RetrainingStatus,
    FeedbackImageDataset,
    get_retrain_manager,
    check_retrain_status,
)

from .comprehensive_trainer import (
    ComprehensiveTrainer,
    ComprehensiveTrainingConfig,
    ComprehensiveTrainingState,
    get_comprehensive_trainer,
    check_comprehensive_training_trigger,
    get_comprehensive_training_status,
)

from .version_manager import (
    ModelVersion,
    ModelVersionManager,
    VersionRecord,
    get_version_manager,
    init_version_manager,
)

from .kd_training import (
    TeacherEnsemble,
    KDConfig,
    KnowledgeDistillationLoss,
    ONNXTeacher,
    PyTorchTeacher,
    create_teacher_ensemble,
    get_teacher_soft_labels,
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
