#!/usr/bin/env python3
"""
Standalone Ensemble Training Script with Super Ensemble Support
Creates separate ensembles for each fusion type, then combines them
INCLUDES: Windows-compatible multiprocessing optimization
FIXED: Cross-attention dimension mismatch and ONNX export issues
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
import numpy as np
import random
import multiprocessing
import platform

# Set multiprocessing method BEFORE any other imports that might use it
if __name__ == "__main__" and platform.system() == 'Windows':
    multiprocessing.set_start_method('spawn', force=True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from tqdm.auto import tqdm

from sklearn.metrics import (precision_score, recall_score, f1_score, 
                              confusion_matrix, accuracy_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Try to import TIMM for DarkNet53
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# ==================== CONFIGURATION ====================
BASE_DIR = Path(r"D:\Base-dir")
DEPLOY_DIR = BASE_DIR / 'deployment_models'
PLOTS_DIR = BASE_DIR / 'plots_metrics'
METRICS_DIR = BASE_DIR / 'metrics_output'
CKPT_DIR = BASE_DIR / 'checkpoints'

# Dataset paths
SPLIT_DIR = Path(r"G:\ML-Model Code\split_dataset")
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"

# Training parameters
ENSEMBLE_INPUT_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# Optimized multiprocessing
def get_optimal_workers():
    """Get optimal number of workers for the current platform"""
    if platform.system() == 'Windows':
        cpu_count = os.cpu_count() or 4
        optimal_workers = max(2, min(8, int(cpu_count * 0.75)))
        return optimal_workers
    else:
        cpu_count = os.cpu_count() or 4
        return max(2, min(12, int(cpu_count * 0.8)))

NUM_WORKERS = get_optimal_workers()

# Super ensemble configuration
TRAIN_SUPER_ENSEMBLE = True  # Set to True to train super ensemble
FUSION_TYPES = ['attention', 'concat', 'cross']  # All three fusion types

# Create directories
for d in [PLOTS_DIR, METRICS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==================== LOGGING ====================
def setup_logger():
    logger = logging.getLogger("ensemble_trainer")
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        fh = logging.FileHandler(BASE_DIR / 'ensemble_training.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

# ==================== REPRODUCIBILITY ====================
def set_seed(seed=SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ==================== WORKER INITIALIZATION ====================
def worker_init_fn(worker_id):
    """Initialize workers properly for Windows multiprocessing"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==================== WINDOWS-COMPATIBLE DATASET ====================

class WindowsCompatibleImageFolder(datasets.ImageFolder):
    """Enhanced ImageFolder optimized for Windows multiprocessing"""
    def __init__(self, root, transform=None, target_transform=None, 
                 loader=datasets.folder.default_loader, is_valid_file=None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        # Pre-cache some metadata for faster access
        self._sample_paths = [s[0] for s in self.samples]
        self._sample_targets = [s[1] for s in self.samples]
    
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            # Return a dummy sample if loading fails
            dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0

# ==================== OPTIMIZED DATALOADER CREATION ====================

def create_optimized_dataloader(dataset, batch_size, shuffle=True, num_workers=None):
    """
    Create optimized DataLoader with proper Windows multiprocessing support
    """
    if num_workers is None:
        num_workers = NUM_WORKERS
    
    # Optimized DataLoader parameters
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else None,
        'worker_init_fn': worker_init_fn if num_workers > 0 else None,
        'drop_last': False,
        'timeout': 30 if num_workers > 0 else 0
    }
    
    # Remove parameters that don't work with num_workers=0
    if num_workers == 0:
        loader_kwargs.pop('persistent_workers', None)
        loader_kwargs.pop('worker_init_fn', None)
        loader_kwargs.pop('timeout', None)
        loader_kwargs.pop('prefetch_factor', None)
        loader_kwargs['pin_memory'] = False
    
    try:
        logger.info(f"Creating DataLoader: {num_workers} workers, batch_size={batch_size}")
        return DataLoader(dataset, **loader_kwargs)
    except Exception as e:
        logger.warning(f"DataLoader creation failed with {num_workers} workers: {e}")
        # Fallback with reduced workers
        fallback_workers = max(1, num_workers // 2) if num_workers > 1 else 0
        loader_kwargs.update({
            'num_workers': fallback_workers,
            'persistent_workers': fallback_workers > 0,
            'worker_init_fn': worker_init_fn if fallback_workers > 0 else None,
            'timeout': 20 if fallback_workers > 0 else 0,
            'prefetch_factor': 2 if fallback_workers > 0 else None
        })
        
        if fallback_workers == 0:
            loader_kwargs.pop('persistent_workers', None)
            loader_kwargs.pop('worker_init_fn', None)
            loader_kwargs.pop('timeout', None)
            loader_kwargs.pop('prefetch_factor', None)
            loader_kwargs['pin_memory'] = False
            
        logger.info(f"Retrying with {fallback_workers} workers")
        return DataLoader(dataset, **loader_kwargs)

# ==================== MODEL ARCHITECTURE ====================

class OptimizedFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, torch.Tensor):
            if out.ndim == 4:
                # Use global average pooling instead of adaptive for ONNX compatibility
                return out.mean(dim=[2, 3])
            elif out.ndim > 2:
                return out.flatten(1)
            return out
        if hasattr(out, 'logits'):
            logits = out.logits
            if isinstance(logits, torch.Tensor):
                if logits.ndim == 4:
                    return logits.mean(dim=[2, 3])
                elif logits.ndim > 2:
                    return logits.flatten(1)
                return logits
        if isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, torch.Tensor):
                    if o.ndim == 4:
                        return o.mean(dim=[2, 3])
                    elif o.ndim > 2:
                        return o.flatten(1)
                    return o
        raise RuntimeError(f"FeatureExtractor couldn't process output")

class OptimizedClassifier(nn.Module):
    def __init__(self, backbone, head, backbone_name):
        super().__init__()
        self.backbone = OptimizedFeatureExtractor(backbone)
        self.head = head
        self.backbone_name = backbone_name
        
    def forward(self, x):
        return self.head(self.backbone(x))
    
    def get_features(self, x):
        """Extract features for ensemble"""
        return self.backbone(x)

# ==================== ENSURE FLOAT32 ====================

def ensure_float32_model(model):
    """Convert all model parameters and buffers to float32"""
    for param in model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()
    
    for buffer in model.buffers():
        if buffer.dtype != torch.float32:
            buffer.data = buffer.data.float()
    
    for module in model.modules():
        if hasattr(module, 'running_mean') and module.running_mean is not None:
            if module.running_mean.dtype != torch.float32:
                module.running_mean = module.running_mean.float()
        if hasattr(module, 'running_var') and module.running_var is not None:
            if module.running_var.dtype != torch.float32:
                module.running_var = module.running_var.float()
    
    return model

# ==================== MODEL LOADING ====================

def detect_feature_dim_from_checkpoint(state_dict):
    """Detect actual feature dimension from saved checkpoint"""
    for key in state_dict.keys():
        if 'head.1.weight' in key:
            return state_dict[key].shape[1]
        if 'head.0.weight' in key and 'head.1.weight' not in state_dict.keys():
            continue
    return None

def create_backbone_and_classifier(backbone_name, num_classes, feat_dim):
    """Create model matching training architecture"""
    
    if backbone_name == 'alexnet':
        backbone = models.alexnet(weights=None)
        backbone.classifier = nn.Identity()
        
    elif backbone_name == 'resnet50':
        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Identity()
        
    elif backbone_name == 'inception_v3':
        backbone = models.inception_v3(weights=None, aux_logits=True)
        backbone.fc = nn.Identity()
        
    elif backbone_name == 'mobilenet_v2':
        backbone = models.mobilenet_v2(weights=None)
        backbone.classifier = nn.Identity()
        
    elif backbone_name == 'efficientnet_b0':
        backbone = models.efficientnet_b0(weights=None)
        backbone.classifier = nn.Identity()
        
    elif 'darknet' in backbone_name:
        if TIMM_AVAILABLE:
            try:
                backbone = timm.create_model('cspresnet50', pretrained=False, num_classes=0)
                logger.info(f"Created CSPResNet50 for DarkNet53")
            except Exception as e:
                logger.warning(f"TIMM failed: {e}, using ResNet50 fallback")
                backbone = models.resnet50(weights=None)
                backbone.fc = nn.Identity()
        else:
            logger.warning("TIMM not available, using ResNet50 fallback")
            backbone = models.resnet50(weights=None)
            backbone.fc = nn.Identity()
            
    elif 'yolo' in backbone_name:
        backbone = models.efficientnet_b0(weights=None)
        backbone.classifier = nn.Identity()
        
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    hidden_dim = max(512, feat_dim // 2)
    classifier_head = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(feat_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, num_classes)
    )
    
    return OptimizedClassifier(backbone, classifier_head, backbone_name)

def load_trained_model(model_id, deployment_dir, num_classes):
    """Load a trained model with proper dtype handling"""
    logger.info(f"Loading {model_id}...")
    
    try:
        deployment_path = Path(deployment_dir)
        model_file = deployment_path / 'model.pth'
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(deployment_path / "class_mapping.json", 'r') as f:
            class_mapping = json.load(f)
        with open(deployment_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        checkpoint = torch.load(model_file, map_location='cpu')
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        
        feat_dim = detect_feature_dim_from_checkpoint(state_dict)
        if feat_dim is None:
            raise ValueError(f"Could not detect feature dimension for {model_id}")
        
        logger.info(f"  Feature dimension: {feat_dim}")
        
        model = create_backbone_and_classifier(model_id, num_classes, feat_dim)
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing:
            logger.warning(f"  Missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"  Unexpected keys: {len(unexpected)}")
        
        model = ensure_float32_model(model)
        
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(2, 3, ENSEMBLE_INPUT_SIZE, ENSEMBLE_INPUT_SIZE)
            test_output = model(test_input)
            logger.info(f"  Verification: Input {test_input.shape} -> Output {test_output.shape}")
        
        logger.info(f"  SUCCESS: {model_id} loaded and verified")
        return model
        
    except Exception as e:
        logger.error(f"  FAILED to load {model_id}: {e}")
        raise

# ==================== BASE ENSEMBLE MODEL ====================

class PestEnsemble(nn.Module):
    """Base ensemble model with dtype safety and cross-attention dimension fix"""
    def __init__(self, models_dict, num_classes, fusion_type='attention'):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        
        self.models = nn.ModuleDict()
        feature_dims = []
        
        for name, model in models_dict.items():
            model = ensure_float32_model(model)
            self.models[name] = model
            
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, ENSEMBLE_INPUT_SIZE, ENSEMBLE_INPUT_SIZE)
                if hasattr(model, 'get_features'):
                    features = model.get_features(dummy_input)
                else:
                    features = model.backbone(dummy_input)
                feature_dims.append(features.shape[1])
        
        self.total_feature_dim = sum(feature_dims)
        logger.info(f"  {fusion_type.upper()} Ensemble - Feature dims: {feature_dims} (total: {self.total_feature_dim})")
        
        if fusion_type == 'attention':
            self.attention_weights = nn.Parameter(torch.ones(len(models_dict)) / len(models_dict))
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.total_feature_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, num_classes)
            )
        elif fusion_type == 'cross':
            # FIX: Project all features to common dimension for cross-attention
            self.common_dim = min(feature_dims)  # Use minimum dimension as common dimension
            
            # Create projection layers for each model to map to common dimension
            self.feature_projections = nn.ModuleList([
                nn.Linear(dim, self.common_dim) for dim in feature_dims
            ])
            
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.common_dim, 
                num_heads=8, 
                batch_first=True
            )
            
            # Fusion layer takes projected features (num_models * common_dim)
            fused_dim = len(models_dict) * self.common_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(fused_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, num_classes)
            )
        else:  # concat
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.total_feature_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        self = ensure_float32_model(self)
    
    def forward(self, x):
        x = x.float()
        
        features_list = []
        
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'get_features'):
                    features = model.get_features(x)
                else:
                    features = model.backbone(x)
                features = features.float()
                features_list.append(features)
        
        if self.fusion_type == 'attention':
            weighted_features = []
            for i, features in enumerate(features_list):
                weighted_features.append(self.attention_weights[i] * features)
            combined_features = torch.cat(weighted_features, dim=1)
            
        elif self.fusion_type == 'cross':
            # FIX: Project all features to common dimension before stacking
            projected_features = []
            for i, features in enumerate(features_list):
                projected = self.feature_projections[i](features)
                projected_features.append(projected)
            
            # Now all features have the same dimension and can be stacked
            stacked_features = torch.stack(projected_features, dim=1)  # [batch, num_models, common_dim]
            
            attended_features, _ = self.cross_attention(
                stacked_features, stacked_features, stacked_features
            )
            combined_features = attended_features.flatten(1)
            
        else:  # concat
            combined_features = torch.cat(features_list, dim=1)
        
        combined_features = combined_features.float()
        
        return self.fusion_layer(combined_features)

# ==================== SUPER ENSEMBLE MODEL ====================

class SuperEnsemble(nn.Module):
    """
    Super Ensemble that combines all three fusion strategies
    Architecture: Three separate ensembles (attention, concat, cross) -> Meta fusion
    """
    def __init__(self, attention_ensemble, concat_ensemble, cross_ensemble, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Store the three ensemble models
        self.attention_ensemble = attention_ensemble
        self.concat_ensemble = concat_ensemble
        self.cross_ensemble = cross_ensemble
        
        # Freeze all base ensembles
        for model in [self.attention_ensemble, self.concat_ensemble, self.cross_ensemble]:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        
        # Meta fusion layer - combines outputs from all three ensembles
        # Input: 3 * num_classes (logits from each ensemble)
        self.meta_fusion = nn.Sequential(
            nn.Linear(3 * num_classes, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Alternative: Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        self = ensure_float32_model(self)
        
        logger.info("Super Ensemble created - combining attention + concat + cross")
    
    def forward(self, x, use_meta_fusion=True):
        """
        Forward pass with two modes:
        - use_meta_fusion=True: Use meta fusion layer (learned combination)
        - use_meta_fusion=False: Use weighted average (simpler)
        """
        x = x.float()
        
        # Get predictions from all three ensembles
        with torch.no_grad():
            attention_logits = self.attention_ensemble(x)
            concat_logits = self.concat_ensemble(x)
            cross_logits = self.cross_ensemble(x)
        
        if use_meta_fusion:
            # Concatenate all logits and pass through meta fusion
            combined_logits = torch.cat([
                attention_logits.float(),
                concat_logits.float(),
                cross_logits.float()
            ], dim=1)
            
            return self.meta_fusion(combined_logits)
        else:
            # Weighted average of logits
            weights = F.softmax(self.ensemble_weights, dim=0)
            return (weights[0] * attention_logits + 
                   weights[1] * concat_logits + 
                   weights[2] * cross_logits)

# ==================== DATA LOADING ====================

def create_transforms(size, is_training=True):
    """Create transforms"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(size * 1.12)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def prepare_datasets():
    """Prepare train and validation datasets with Windows-compatible loaders"""
    train_transform = create_transforms(ENSEMBLE_INPUT_SIZE, is_training=True)
    val_transform = create_transforms(ENSEMBLE_INPUT_SIZE, is_training=False)
    
    train_ds = WindowsCompatibleImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_ds = WindowsCompatibleImageFolder(str(VAL_DIR), transform=val_transform)
    
    logger.info(f"Dataset loaded: Train={len(train_ds)}, Val={len(val_ds)}, Classes={len(train_ds.classes)}")
    
    return train_ds, val_ds

# ==================== TRAINING ====================

def train_epoch(model, loader, optimizer, criterion, is_super_ensemble=False):
    """Train for one epoch"""
    model.train()
    
    # For base ensembles, keep backbone models in eval
    if not is_super_ensemble and hasattr(model, 'models'):
        for name, backbone_model in model.models.items():
            backbone_model.eval()
    
    # For super ensemble, keep base ensembles in eval
    if is_super_ensemble:
        model.attention_ensemble.eval()
        model.concat_ensemble.eval()
        model.cross_ensemble.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, targets in pbar:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(targets.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, acc, f1

def validate_epoch(model, loader, criterion):
    """Validate for one epoch"""
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, acc, prec, rec, f1, all_preds, all_labels, all_probs

# ==================== MODEL EXPORT ====================

def export_model_to_onnx(model, model_name, input_size, save_dir):
    """Export model to ONNX format with dtype fix"""
    try:
        export_model = ensure_float32_model(model)
        export_model.eval()
        export_model = export_model.cpu()
        
        dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
        
        onnx_path = Path(save_dir) / f"{model_name}.onnx"
        
        with torch.no_grad():
            torch.onnx.export(
                export_model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                verbose=False
            )
        
        logger.info(f"  ONNX exported: {onnx_path}")
        return str(onnx_path)
        
    except Exception as e:
        logger.error(f"  ONNX export failed: {e}")
        return None

def export_model_to_torchscript(model, model_name, input_size, save_dir):
    """Export model to TorchScript with dtype fix"""
    try:
        export_model = ensure_float32_model(model)
        export_model.eval()
        export_model = export_model.cpu()
        
        dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
        
        with torch.no_grad():
            traced_model = torch.jit.trace(export_model, dummy_input, strict=False)
        
        torchscript_path = Path(save_dir) / f"{model_name}.pt"
        traced_model.save(str(torchscript_path))
        
        logger.info(f"  TorchScript exported: {torchscript_path}")
        return str(torchscript_path)
        
    except Exception as e:
        logger.error(f"  TorchScript export failed: {e}")
        return None

def export_model_safe(model, model_name, input_size, save_dir):
    """Safe model export with all formats"""
    logger.info(f"Exporting {model_name} for deployment...")
    
    try:
        original_device = next(model.parameters()).device
        
        export_results = {
            'onnx_path': None,
            'torchscript_path': None,
            'pytorch_path': None
        }
        
        # ONNX export
        onnx_path = export_model_to_onnx(model, model_name, input_size, save_dir)
        export_results['onnx_path'] = onnx_path
        
        # TorchScript export
        torchscript_path = export_model_to_torchscript(model, model_name, input_size, save_dir)
        export_results['torchscript_path'] = torchscript_path
        
        # PyTorch state dict
        try:
            pytorch_path = Path(save_dir) / f"{model_name}_state_dict.pth"
            model_cpu = model.cpu()
            model_cpu = ensure_float32_model(model_cpu)
            
            torch.save({
                'model_state_dict': model_cpu.state_dict(),
                'model_class': model.__class__.__name__,
                'input_size': input_size,
                'dtype': 'float32'
            }, str(pytorch_path))
            export_results['pytorch_path'] = str(pytorch_path)
            logger.info(f"  PyTorch state dict saved: {pytorch_path}")
        except Exception as pt_e:
            logger.warning(f"  PyTorch save failed: {pt_e}")
        
        # Restore model to original device
        model.to(original_device)
        
        return export_results
        
    except Exception as e:
        logger.error(f"Export failed for {model_name}: {e}")
        return {'onnx_path': None, 'torchscript_path': None, 'pytorch_path': None}

def save_deployment_package(model, model_name, class_names, metrics, save_dir, export_formats=True):
    """Save complete deployment package with all export formats"""
    package_dir = Path(save_dir) / f"{model_name}_deployment"
    package_dir.mkdir(exist_ok=True)
    
    # Save primary PyTorch model
    model_path = package_dir / "model.pth"
    model_cpu = model.cpu()
    model_cpu = ensure_float32_model(model_cpu)
    torch.save(model_cpu.state_dict(), model_path)
    logger.info(f"Model saved: {model_path}")
    
    # Export to multiple formats if requested
    export_results = {}
    if export_formats:
        export_results = export_model_safe(model, model_name, ENSEMBLE_INPUT_SIZE, package_dir)
        metrics['exports'] = export_results
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'class_names': class_names,
        'num_classes': len(class_names),
        'metrics': metrics,
        'input_size': ENSEMBLE_INPUT_SIZE,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pytorch_version': torch.__version__,
    }
    
    metadata_path = package_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save class mapping
    class_mapping = {i: name for i, name in enumerate(class_names)}
    mapping_path = package_dir / "class_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    logger.info(f"Deployment package complete: {package_dir}")
    
    # Log export status
    if export_formats:
        onnx_status = "cool" if export_results.get('onnx_path') else "x"
        ts_status = "cool" if export_results.get('torchscript_path') else "x"
        pt_status = "cool" if export_results.get('pytorch_path') else "x"
        logger.info(f"  Exports: ONNX {onnx_status} | TorchScript {ts_status} | PyTorch {pt_status}")
    
    return str(package_dir)

# ==================== VISUALIZATION ====================

def plot_training_history(history, save_path, title="Training History"):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    epochs = list(range(len(history['train_loss'])))
    
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', color='blue')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['train_acc'], label='Train', color='blue')
    axes[0, 1].plot(epochs, history['val_acc'], label='Val', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['val_f1'], label='Val F1', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].text(0.5, 0.5, 
                    f'Best Val Acc: {max(history["val_acc"]):.4f}\n' +
                    f'Best Val F1: {max(history["val_f1"]):.4f}\n' +
                    f'Final Train Loss: {history["train_loss"][-1]:.4f}\n' +
                    f'Final Val Loss: {history["val_loss"][-1]:.4f}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1, 1].transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_title('Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training history saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """Plot confusion matrix"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f'{title} - Raw Counts')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title(f'{title} - Normalized')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")

def plot_roc_curves(y_true, y_probs, class_names, save_path, title="ROC Curves"):
    """Plot ROC curves"""
    plt.figure(figsize=(12, 10))
    
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    roc_auc = {}
    for i in range(len(class_names)):
        if y_true_bin.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        else:
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, i])
        roc_auc[i] = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curves saved: {save_path}")
    
    return roc_auc

# ==================== TRAINING FUNCTIONS ====================

def train_single_ensemble(trained_models, train_loader, val_loader, class_names, 
                         num_classes, fusion_type):
    """Train a single ensemble with specific fusion type"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {fusion_type.upper()} Ensemble")
    logger.info(f"{'='*60}")
    
    # Create ensemble
    ensemble = PestEnsemble(trained_models, num_classes, fusion_type)
    ensemble = ensemble.to(DEVICE)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(ensemble.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    best_acc = 0.0
    best_state = None
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    # Training loop
    for epoch in range(EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc, train_f1 = train_epoch(
            ensemble, train_loader, optimizer, criterion
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels, val_probs = validate_epoch(
            ensemble, val_loader, criterion
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = ensemble.state_dict().copy()
            logger.info(f"  NEW BEST: {best_acc:.4f}")
        
        logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
    
    # Load best model
    if best_state is not None:
        ensemble.load_state_dict(best_state)
    
    # Final evaluation
    val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels, val_probs = validate_epoch(
        ensemble, val_loader, criterion
    )
    
    # Generate visualizations
    plot_training_history(history, 
                         PLOTS_DIR / f"ensemble_{fusion_type}_training_history.png",
                         f"{fusion_type.upper()} Ensemble Training")
    plot_confusion_matrix(val_labels, val_preds, class_names,
                         PLOTS_DIR / f"ensemble_{fusion_type}_confusion_matrix.png",
                         f"{fusion_type.upper()} Ensemble")
    roc_auc = plot_roc_curves(val_labels, val_probs, class_names,
                              PLOTS_DIR / f"ensemble_{fusion_type}_roc_curves.png",
                              f"{fusion_type.upper()} Ensemble ROC")
    
    # Save metrics
    metrics = {
        'accuracy': float(best_acc),
        'precision': float(val_prec),
        'recall': float(val_rec),
        'f1_score': float(val_f1),
        'final_val_loss': float(val_loss),
        'roc_auc_scores': {class_names[i]: float(score) for i, score in roc_auc.items()},
        'macro_roc_auc': float(np.mean(list(roc_auc.values()))),
        'fusion_type': fusion_type,
        'num_models': len(trained_models),
        'model_names': list(trained_models.keys()),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save checkpoint
    checkpoint_path = CKPT_DIR / f'ensemble_{fusion_type}.pth'
    torch.save({
        'model_state_dict': ensemble.state_dict(),
        'extra': metrics
    }, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save deployment package with all export formats
    model_name = f'ensemble_{fusion_type}'
    deployment_package = save_deployment_package(
        ensemble, model_name, class_names, metrics, DEPLOY_DIR, export_formats=True
    )
    
    # Save detailed metrics
    metrics_file = METRICS_DIR / f"{model_name}_detailed_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved: {metrics_file}")
    
    logger.info(f"\n{fusion_type.upper()} Ensemble Complete: Acc={best_acc:.4f}, F1={val_f1:.4f}")
    
    return ensemble, best_acc, metrics

def train_super_ensemble(attention_ensemble, concat_ensemble, cross_ensemble,
                        train_loader, val_loader, class_names, num_classes):
    """Train the super ensemble that combines all three fusion types"""
    logger.info(f"\n{'='*60}")
    logger.info("Training SUPER ENSEMBLE (Meta-Fusion)")
    logger.info(f"{'='*60}")
    
    # Create super ensemble
    super_ensemble = SuperEnsemble(attention_ensemble, concat_ensemble, cross_ensemble, num_classes)
    super_ensemble = super_ensemble.to(DEVICE)
    
    # Training setup - only train meta fusion layer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(super_ensemble.meta_fusion.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    best_acc = 0.0
    best_state = None
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    # Training loop
    for epoch in range(EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc, train_f1 = train_epoch(
            super_ensemble, train_loader, optimizer, criterion, is_super_ensemble=True
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels, val_probs = validate_epoch(
            super_ensemble, val_loader, criterion
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = super_ensemble.state_dict().copy()
            logger.info(f"  NEW BEST: {best_acc:.4f}")
        
        logger.info(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")
        logger.info(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
    
    # Load best model
    if best_state is not None:
        super_ensemble.load_state_dict(best_state)
    
    # Final evaluation
    val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels, val_probs = validate_epoch(
        super_ensemble, val_loader, criterion
    )
    
    # Generate visualizations
    plot_training_history(history,
                         PLOTS_DIR / "super_ensemble_training_history.png",
                         "Super Ensemble Training (Meta-Fusion)")
    plot_confusion_matrix(val_labels, val_preds, class_names,
                         PLOTS_DIR / "super_ensemble_confusion_matrix.png",
                         "Super Ensemble")
    roc_auc = plot_roc_curves(val_labels, val_probs, class_names,
                              PLOTS_DIR / "super_ensemble_roc_curves.png",
                              "Super Ensemble ROC")
    
    # Save metrics
    metrics = {
        'accuracy': float(best_acc),
        'precision': float(val_prec),
        'recall': float(val_rec),
        'f1_score': float(val_f1),
        'final_val_loss': float(val_loss),
        'roc_auc_scores': {class_names[i]: float(score) for i, score in roc_auc.items()},
        'macro_roc_auc': float(np.mean(list(roc_auc.values()))),
        'fusion_type': 'super_ensemble_meta_fusion',
        'component_ensembles': ['attention', 'concat', 'cross'],
        'architecture': 'hierarchical_ensemble',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save checkpoint
    checkpoint_path = CKPT_DIR / 'super_ensemble.pth'
    torch.save({
        'model_state_dict': super_ensemble.state_dict(),
        'extra': metrics
    }, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save deployment package with all export formats
    model_name = 'super_ensemble'
    deployment_package = save_deployment_package(
        super_ensemble, model_name, class_names, metrics, DEPLOY_DIR, export_formats=True
    )
    
    # Save detailed metrics
    metrics_file = METRICS_DIR / "super_ensemble_detailed_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved: {metrics_file}")
    
    logger.info(f"\nSuper Ensemble Complete: Acc={best_acc:.4f}, F1={val_f1:.4f}")
    
    return super_ensemble, best_acc, metrics

# ==================== MAIN ====================

def main():
    logger.info("="*60)
    logger.info("ENSEMBLE TRAINING WITH SUPER ENSEMBLE SUPPORT")
    logger.info("Optimized Windows Multiprocessing Enabled")
    logger.info("="*60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Platform: {platform.system()}")
    logger.info(f"CPU Cores: {os.cpu_count()}")
    logger.info(f"Optimal Workers: {NUM_WORKERS}")
    logger.info(f"TIMM Available: {TIMM_AVAILABLE}")
    logger.info(f"Fusion Types: {FUSION_TYPES}")
    logger.info(f"Super Ensemble: {TRAIN_SUPER_ENSEMBLE}")
    logger.info(f"Seed: {SEED}")
    logger.info("")
    
    # Find available models
    available_models = []
    for model_id in ['alexnet', 'resnet50', 'inception_v3', 'mobilenet_v2', 
                     'efficientnet_b0', 'darknet53', 'yolo11n-cls']:
        deployment_dir = DEPLOY_DIR / f"{model_id}_deployment"
        if deployment_dir.exists():
            files = ['class_mapping.json', 'metadata.json', 'model.pth']
            if all((deployment_dir / f).exists() for f in files):
                available_models.append((model_id, deployment_dir))
    
    if len(available_models) < 2:
        logger.error(f"Not enough models found: {len(available_models)}")
        return
    
    logger.info(f"Found {len(available_models)} models:")
    for model_id, _ in available_models:
        logger.info(f"  - {model_id}")
    logger.info("")
    
    # Prepare datasets
    train_ds, val_ds = prepare_datasets()
    num_classes = len(train_ds.classes)
    class_names = train_ds.classes
    
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Classes: {num_classes}\n")
    
    # Load models
    logger.info("Loading pre-trained models...")
    trained_models = {}
    
    for model_id, deployment_dir in available_models:
        try:
            model = load_trained_model(model_id, deployment_dir, num_classes)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            trained_models[model_id] = model
        except Exception as e:
            logger.error(f"Failed to load {model_id}: {e}")
    
    if len(trained_models) < 2:
        logger.error("Not enough models loaded")
        return
    
    logger.info(f"\nLoaded {len(trained_models)} models\n")
    
    # Create optimized dataloaders with multiprocessing
    logger.info(f"Creating dataloaders with {NUM_WORKERS} workers...")
    train_loader = create_optimized_dataloader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = create_optimized_dataloader(val_ds, BATCH_SIZE, shuffle=False)
    logger.info("Dataloaders created successfully\n")
    
    # Train individual ensembles
    ensemble_results = {}
    trained_ensembles = {}
    
    for fusion_type in FUSION_TYPES:
        ensemble, acc, metrics = train_single_ensemble(
            trained_models, train_loader, val_loader, 
            class_names, num_classes, fusion_type
        )
        trained_ensembles[fusion_type] = ensemble
        ensemble_results[fusion_type] = {'accuracy': acc, 'metrics': metrics}
    
    # Train super ensemble
    if TRAIN_SUPER_ENSEMBLE and len(trained_ensembles) == 3:
        super_ensemble, super_acc, super_metrics = train_super_ensemble(
            trained_ensembles['attention'],
            trained_ensembles['concat'],
            trained_ensembles['cross'],
            train_loader, val_loader, class_names, num_classes
        )
        ensemble_results['super'] = {'accuracy': super_acc, 'metrics': super_metrics}
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - RESULTS SUMMARY")
    logger.info("="*60)
    
    for name, result in ensemble_results.items():
        logger.info(f"{name.upper():15s} | Accuracy: {result['accuracy']:.4f}")
    
    logger.info("\n" + "-"*60)
    logger.info("DEPLOYMENT EXPORTS")
    logger.info("-"*60)
    
    for name, result in ensemble_results.items():
        if 'metrics' in result and 'exports' in result['metrics']:
            exports = result['metrics']['exports']
            onnx = "cool" if exports.get('onnx_path') else "x"
            ts = "cool" if exports.get('torchscript_path') else "x"
            pt = "cool" if exports.get('pytorch_path') else "x"
            logger.info(f"{name.upper():15s} | ONNX {onnx} | TorchScript {ts} | PyTorch {pt}")
    
    logger.info("="*60)
    logger.info(f"All models saved to: {DEPLOY_DIR}")
    logger.info(f"Visualizations saved to: {PLOTS_DIR}")
    logger.info(f"Metrics saved to: {METRICS_DIR}")
    logger.info(f"Checkpoints saved to: {CKPT_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise