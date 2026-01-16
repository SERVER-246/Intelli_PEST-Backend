# Intelli-PEST Project Journey
## Complete Technical Documentation: From Initial Training to Production Deployment

**Document Version:** 1.3  
**Last Updated:** January 16, 2026 (Phase 3 Fully Operational)  
**Project:** Sugarcane Pest Detection using Deep Learning  
**Status:** Production (Server-Side Inference with Phase 3 Features)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Timeline Overview](#2-project-timeline-overview)
3. [Phase 1: Initial Model Training & ONNX Models](#3-phase-1-initial-model-training--onnx-models)
4. [Phase 2: TFLite Conversion Attempt](#4-phase-2-tflite-conversion-attempt)
5. [Phase 3: TFLite Compatibility Crisis](#5-phase-3-tflite-compatibility-crisis)
6. [Phase 4: The Decision to Move to Server-Side Inference](#6-phase-4-the-decision-to-move-to-server-side-inference)
7. [Phase 5: Server Infrastructure Development](#7-phase-5-server-infrastructure-development)
8. [Phase 6: Android Application Refactoring](#8-phase-6-android-application-refactoring)
9. [Phase 7: Model Improvements & Version History](#9-phase-7-model-improvements--version-history)
10. [Phase 8: Production System & Auto-Retraining](#10-phase-8-production-system--auto-retraining)
11. [Phase 9: Black-Ops Training Phase 2](#11-phase-9-black-ops-training-phase-2)
12. [Phase 10: Phase 3 Implementation (NEW)](#12-phase-10-phase-3-implementation-new)
13. [Issues & Bugs Encountered](#13-issues--bugs-encountered)
14. [Current State & Architecture](#14-current-state--architecture)
15. [Lessons Learned](#15-lessons-learned)

---

## 1. Executive Summary

The Intelli-PEST project is an AI-powered sugarcane pest detection system that went through significant architectural evolution. What started as an **on-device TFLite inference** mobile application transformed into a **server-side PyTorch inference** system due to critical compatibility issues with TensorFlow Lite on Android devices.

### Key Achievements
- **Final Model Accuracy:** 96.25% validation accuracy
- **Classes Supported:** 12 (11 pest types + 1 junk/filter class)
- **Architecture:** Enhanced CNN with CBAM, FPN, Knowledge Consolidation
- **Model Size:** ~46.85 MB (PyTorch)
- **Teacher Models Used:** 11 (7 base + 4 ensemble)
- **Current Version:** v1.0.5

### Major Pivots
1. ONNX → TFLite conversion **failed** due to operator version incompatibility
2. Moved from on-device to **server-side inference** via VS Code Dev Tunnels
3. Expanded from 11 classes to **12 classes** (added junk filter)
4. Implemented **automatic retraining** with user feedback

---

## 2. Project Timeline Overview

| Date | Phase | Key Event |
|------|-------|-----------|
| **Dec 2024** | Initial Training | 7 base models trained, 4 ensemble models created |
| **Dec 10, 2025** | v1.0.0 App Release | Initial app with local ONNX model inference |
| **Dec 12, 2025** | TFLite Conversion | First conversion attempt - 5/11 models succeeded |
| **Dec 15-18, 2025** | Compatibility Crisis | FULLY_CONNECTED version 12 error discovered |
| **Dec 18, 2025** | Pivot Decision | Decided to move to server-side inference |
| **Dec 20-23, 2025** | Knowledge Distillation | Sequential training from all 11 teachers |
| **Dec 31, 2025** | v1.1.0 App Release | Server-side inference, Firebase Auth, GPS tracking |
| **Jan 1, 2026** | Rotation Training | 360° rotation-robust training initiated |
| **Jan 9, 2026** | 12-Class Expansion | Added junk class for filtering |
| **Jan 9-14, 2026** | Production Refinement | Auto-retraining, bug fixes, stability improvements |
| **Jan 14, 2026** | Phase 2 Complete | v1.0.5 deployed, auto-scheduler active, 97 tests passing |
| **Jan 15, 2026** | Phase 3 Planning | Region-aware perception readiness audit completed |
| **Jan 16, 2026** | Phase 3 Implementation | Phase 3 APIs, app version system fix, model path fix |
| **Jan 16, 2026** | Phase 3 Fully Operational | Region scoring, multi-label classification working |

---

## 3. Phase 1: Initial Model Training & ONNX Models

### 3.1 Dataset Preparation

The project started with a curated dataset of sugarcane pest images:

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 3,067 | 80% |
| Validation | 773 | 20% |
| **Total** | **3,840** | 100% |

**11 Original Classes:**
1. Healthy
2. Internode borer
3. Pink borer
4. Rat damage
5. Stalk borer
6. Top borer
7. Army worm
8. Mealy bug
9. Porcupine damage
10. Root borer
11. Termite

### 3.2 Base Model Training

Seven architectures were trained using transfer learning from ImageNet:

| Model | Architecture | Fine-tuned Accuracy |
|-------|-------------|---------------------|
| AlexNet | Classic CNN | ~75% |
| MobileNetV2 | Efficient mobile | ~85% |
| EfficientNet-B0 | Compound scaling | ~88% |
| ResNet50 | Residual connections | ~87% |
| DarkNet53 | YOLO backbone | ~86% |
| InceptionV3 | Multi-scale features | ~86% |
| YOLO11n-cls | Ultralytics classifier | ~84% |

**Training Configuration:**
- Epochs: 40 (head) + 25 (fine-tuning)
- Batch Size: 32
- Learning Rate: 0.001 with CosineAnnealing
- Optimizer: AdamW
- Augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter, Rotation

### 3.3 Ensemble Model Creation

Four ensemble models were created by combining the base models:

| Ensemble | Fusion Method | Accuracy |
|----------|--------------|----------|
| Ensemble_Attention | Learned attention weights | ~92% |
| Ensemble_Concat | Feature concatenation + reduction | ~91% |
| Ensemble_Cross | Cross-attention between models | ~90% |
| Super_Ensemble | Meta-ensemble of all 3 | ~95% |

### 3.4 ONNX Conversion

All 11 models were successfully converted to ONNX format:

**Conversion Process:**
- PyTorch → ONNX using `torch.onnx.export()`
- Opset version: 13
- Dynamic axes enabled for batch size flexibility

**ONNX Model Sizes:**

| Model | ONNX Size |
|-------|-----------|
| alexnet.onnx | 222.24 MB |
| mobilenet_v2.onnx | 3.47 MB |
| efficientnet_b0.onnx | 5.31 MB |
| resnet50.onnx | 24.51 MB |
| darknet53.onnx | 20.24 MB |
| inception_v3.onnx | 23.35 MB |
| yolo11n-cls.onnx | 5.33 MB |
| ensemble_attention.onnx | 99.35 MB |
| ensemble_concat.onnx | 99.81 MB |
| ensemble_cross.onnx | 106.87 MB |
| super_ensemble.onnx | 144.54 MB |

---

## 4. Phase 2: TFLite Conversion Attempt

### 4.1 Why TFLite?

The original plan was to deploy models on Android devices for **offline inference**. TensorFlow Lite was chosen because:
- Native Android support
- Optimized for mobile hardware
- GPU/NNAPI delegate support
- Smaller model footprint (quantization)

### 4.2 Conversion Pipeline

The planned pipeline was:
```
PyTorch (.pt) → ONNX (.onnx) → TensorFlow SavedModel → TFLite (.tflite)
```

### 4.3 First Conversion Attempt (Dec 12, 2025)

**Tools Used:**
- TensorFlow 2.15.0
- onnx-tf (ONNX to TensorFlow converter)
- Dynamic range quantization

**Results:**

| Model | Status | Issue |
|-------|--------|-------|
| alexnet | ✅ Success | - |
| mobilenet_v2 | ✅ Success | - |
| efficientnet_b0 | ✅ Success | - |
| resnet50 | ✅ Success | - |
| inception_v3 | ✅ Success | - |
| darknet53 | ❌ Failed | KeyError: 'split' |
| yolo11n-cls | ❌ Failed | Unsupported ops |
| ensemble_attention | ❌ Failed | Complex attention layers |
| ensemble_concat | ❌ Failed | Concat dimension mismatch |
| ensemble_cross | ❌ Failed | Cross-attention not supported |
| super_ensemble | ❌ Failed | Meta-ensemble complexity |

**Success Rate:** 5/11 (45.5%)

### 4.4 Issues Encountered During Conversion

**Issue 1: KeyError 'split' in DarkNet53**
- The ONNX model used Split operation with parameters the converter couldn't handle
- Error: `KeyError: 'split'` in onnx2keras converter

**Issue 2: Unsupported Operations**
- Custom layers in ensemble models not supported by TFLite
- Attention mechanisms required SELECT_TF_OPS delegate

**Issue 3: UnicodeEncodeError on Windows**
- Arrow characters (→) in print statements caused encoding errors
- Error: `'charmap' codec can't encode character '\u2192'`
- Fix: Forced UTF-8 encoding or replaced Unicode characters

---

## 5. Phase 3: TFLite Compatibility Crisis

### 5.1 The Critical Error

After successfully converting 5 models to TFLite, testing on Android revealed a **critical compatibility error**:

```
Didn't find op for builtin opcode 'FULLY_CONNECTED' version '12'
```

### 5.2 Root Cause Analysis

**Problem:** TensorFlow 2.20 (used for conversion) produced TFLite models using **newer operation versions** that older Android TFLite runtimes couldn't interpret.

**Technical Details:**
- TF 2.20 converter uses `FULLY_CONNECTED version 12`
- Android TFLite Runtime 2.14.0-2.16.0 only supports up to `version 9`
- This is a **forward compatibility issue** - newer models on older runtimes

### 5.3 Compatibility Matrix

| TFLite Runtime | tflite_models/ | Status |
|----------------|----------------|--------|
| 2.14.0 | ❌ FAILED | Op version too new |
| 2.15.0 | ❌ FAILED | Op version too new |
| 2.16.0 | ❌ FAILED | Op version too new |
| 2.16.1 | ⚠️ Partial | Some ops work |
| 2.17.0+ | ✅ Works | Runtime updated |

### 5.4 Attempted Fixes

**Attempt 1: Downgrade TensorFlow for conversion**
- Used TF 2.14.0 for conversion
- Result: Some models worked, but ensemble models still failed

**Attempt 2: Use older ONNX opset**
- Converted with opset 11 instead of 13
- Result: Improved compatibility but lost some operations

**Attempt 3: Legacy conversion scripts**
- Created `convert_compatible.py` and `convert_legacy.py`
- Used strictest backward compatibility settings
- Result: Better compatibility but model functionality issues

**Attempt 4: Flex Delegate**
- Added SELECT_TF_OPS for unsupported operations
- Result: Required additional dependencies, increased app size, still unstable

### 5.5 The Fundamental Problem

Even with compatible models, the following issues remained:
1. **Limited model support** - Only 5 of 11 models could convert
2. **No ensemble inference** - Best models (ensembles) couldn't be used
3. **Input size variations** - Different models needed different input sizes (224x224, 256x256, 299x299)
4. **Ongoing maintenance burden** - Each TFLite runtime update could break compatibility

---

## 6. Phase 4: The Decision to Move to Server-Side Inference

### 6.1 Why Server-Side?

After weeks of battling TFLite compatibility issues, the team made a strategic decision to move to server-side inference:

**Advantages:**
1. **Full model support** - All 11 teachers available, plus enhanced student model
2. **No compatibility issues** - PyTorch runs consistently on server
3. **Easy updates** - Model updates don't require app updates
4. **More compute power** - GPU acceleration on server
5. **Unified codebase** - Same inference code for all platforms

**Trade-offs:**
1. **Internet required** - No offline detection
2. **Latency** - Network round-trip added
3. **Server costs** - Need to maintain server infrastructure
4. **Privacy concerns** - Images sent to server

### 6.2 Architecture Decision

**Chosen Stack:**
- **Server:** FastAPI with Uvicorn
- **Model Format:** PyTorch (.pt)
- **Tunneling:** VS Code Dev Tunnels (for development)
- **API Design:** RESTful with JSON responses
- **Security:** API key authentication

---

## 7. Phase 5: Server Infrastructure Development

### 7.1 Knowledge Distillation Training

To create an optimal model for server deployment, **sequential knowledge distillation** was performed:

**Configuration:**
- 11 teacher models (7 base + 4 ensemble)
- 20 epochs per teacher = 220 epochs
- 30 epochs final refinement
- Total: 250 epochs (~22 hours)
- Temperature: 4.0
- Alpha (hard labels): 0.3
- Beta (soft labels): 0.7

**Student Model Architecture:**
- Enhanced CNN with CBAM (attention)
- FPN (Feature Pyramid Network)
- Knowledge Consolidation blocks
- Depthwise separable convolutions
- 12.2M parameters
- 46.85 MB size

**Per-Teacher Results:**

| Teacher | Best Accuracy | Final Val Acc |
|---------|---------------|---------------|
| alexnet | 79.30% | 79.30% |
| mobilenet_v2 | 86.42% | 83.31% |
| efficientnet_b0 | 88.23% | 84.22% |
| resnet50 | 90.43% | 86.68% |
| darknet53 | 91.33% | 90.43% |
| inception_v3 | 92.11% | 90.17% |
| yolo11n-cls | 93.79% | 91.98% |
| ensemble_attention | 94.83% | 92.37% |
| ensemble_concat | 93.79% | 92.76% |
| ensemble_cross | 50.19%* | 45.41%* |
| super_ensemble | 92.24% | 86.03% |

*ensemble_cross had training issues and produced poor results

**Final Result:** **96.25% validation accuracy**

### 7.2 Inference Server Development

**Server Components:**

1. **FastAPI Application** (`inference_server/fastapi_app/`)
   - Main entry point
   - API routers
   - Request/response schemas
   - Dependency injection

2. **Engine** (`inference_server/engine/`)
   - Model loading
   - PyTorch inference
   - ONNX inference (backup)
   - Model registry

3. **Security** (`inference_server/security/`)
   - API key management
   - Rate limiting
   - Input sanitization
   - Audit logging

4. **Filters** (`inference_server/filters/`)
   - File validation
   - Image validation
   - Content filtering
   - Out-of-distribution detection

5. **Feedback System** (`inference_server/feedback/`)
   - User tracking
   - Data collection
   - Database management (SQLite)
   - Image storage

6. **Training System** (`inference_server/training/`)
   - Retrain manager
   - Auto-scheduler
   - Comprehensive trainer

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/classes` | GET | List pest classes |
| `/api/v1/models` | GET | List available models |
| `/api/v1/predict` | POST | Single image prediction |
| `/api/v1/predict/base64` | POST | Base64 image prediction |
| `/api/v1/feedback` | POST | Submit user feedback |

### 7.3 Server Configuration

**Uvicorn Settings:**
- Host: 0.0.0.0
- Port: 8000
- Concurrent connections: 100+
- Connection timeout: 60s
- Backlog: 2048
- Loop: asyncio

**VS Code Dev Tunnel:**
- URL: `https://cnbrr7xn-8000.inc1.devtunnels.ms`
- Protocol: HTTPS (encrypted)
- Authentication: VS Code Dev Tunnels auth

---

## 8. Phase 6: Android Application Refactoring

### 8.1 From Local to Remote Inference

**v1.0.0 Architecture (Local):**
```
Camera/Gallery → Image → TFLite Model → Prediction → Display
```

**v1.1.0 Architecture (Remote):**
```
Camera/Gallery → Image → API Request → Server → API Response → Display
```

### 8.2 Major App Changes

1. **Removed:** TFLite dependencies and model assets
2. **Added:** Retrofit2 for HTTP requests
3. **Added:** Firebase Authentication
4. **Added:** Location service (GPS tracking)
5. **Added:** Room database for history
6. **Refactored:** Repository pattern for API calls

### 8.3 API Configuration

**ApiConfig.kt:**
- Dev Tunnel URL for real devices
- Emulator URL (10.0.2.2) for testing
- Timeouts: Connect 90s, Read 120s, Write 90s
- Max retries: 3
- API Key header: `X-API-Key`

### 8.4 New Features Added

1. **Firebase Email/Password Auth** - User registration and login
2. **Google Sign-In** - OAuth integration
3. **Email Verification** - Required for new users
4. **GPS Location Tracking** - Background service with foreground notification
5. **Detection History** - Local storage with Room database
6. **Statistics** - Average confidence, common pests
7. **Network Monitoring** - No-network screen

### 8.5 App Issues Fixed

**Issue 1: Timeout Errors**
- Original timeouts too short for server inference
- Fix: Increased to 90s connect, 120s read

**Issue 2: Emulator Connection**
- Emulator couldn't reach Dev Tunnel URL
- Fix: Added special handling for 10.0.2.2 (host localhost)

**Issue 3: Image Upload Size**
- Large images caused slow uploads
- Fix: Image compression before upload

**Issue 4: Bitmap Configuration Crashes**
- Hardware bitmaps couldn't be processed
- Fix: Conversion to software bitmaps

---

## 9. Phase 7: Model Improvements & Version History

### 9.1 Rotation Robustness Training

**Problem:** Model performed poorly on rotated images in real-world usage.

**Solution:** Fine-tuning with 360° rotation augmentation.

**Training Configuration:**
- 24 rotation angles (0°, 15°, 30°, ... 345°)
- 80% rotation probability
- 150 epochs
- All 11 teachers + deployed model as 12th teacher

**Result:** `student_model_rotation_robust.pt` created

### 9.2 12-Class Expansion

**Problem:** Users submitting non-pest images (food, selfies, random objects) that received high-confidence pest predictions.

**Solution:** Added **"junk" class** as 12th class to filter irrelevant images.

**Implementation Challenge:**
The initial attempt to expand the classifier broke the model:

**Broken Approach (v1.0.1):**
- Expanded only main classifier
- Forgot `aux_classifiers` in EnhancedStudentModel
- Result: `aux_classifiers` mismatch error

**Proper Approach:**
- Expanded BOTH main classifier AND aux_classifiers
- Conservative initialization for new class (small weights, negative bias)
- Preserved original 11-class weights exactly

**Script:** `expand_model_properly.py`
**Output:** `student_model_12class_proper.pt`

### 9.3 Model Version History

| Version | Model | Changes |
|---------|-------|---------|
| v1.0.0 | student_model_final.pth | Original 11-class, 96.25% accuracy |
| v1.0.1 | student_model_v1.0.1.pt | **BROKEN** - aux_classifiers mismatch |
| v1.0.2 | student_model_v1.0.2.pt | Fine-tuning attempt |
| v1.0.3 | student_model_v1.0.3.pt | Fine-tuning attempt |
| v1.0.4 | student_model_v1.0.4.pt | Fine-tuning attempt |
| v1.0.5 | student_model_12class_proper.pt | **CURRENT** - Properly expanded 12-class |

### 9.4 Black-Ops Ghost Training

**Purpose:** Silent comprehensive training that improves the model without updating official records.

**Features:**
- Sequential KD with 12 teachers
- Full rotation augmentation
- Mixup and CutMix augmentation
- Gradient clipping for stability
- Checkpoint recovery

**Key Difference:** Ghost training does NOT:
- Change version number
- Update retrain_status.json
- Write to database
- Archive feedback images
- Push to Git

### 9.5 Black-Ops Training System - Phase 2 Enhancements (Jan 14, 2026)

A comprehensive upgrade to the ghost training system replacing unstable components with production-ready alternatives.

#### Phase 2 Components

| Component | Replaces | Purpose |
|-----------|----------|---------|
| **LwF (Learning without Forgetting)** | EWC | Soft target matching to preserve old knowledge |
| **Memory Replay Buffer** | - | Reservoir sampling of 1000 exemplars |
| **Soft NaN Regulation** | Hard disable | Graceful feature degradation (1.0 → 0.5 → 0.2 → disable) |
| **Multi-Format Reports** | Single JSON | JSON + Markdown + Interactive HTML |
| **Run Archival** | Manual | Auto-archive to `runs/<timestamp>/` |
| **Auto-Deployment** | Manual | Deploy with backup, verification, and rollback |
| **Experiment Tracking** | - | Persistent history with best-run tracking |
| **Batch Probe Caching** | - | Hardware-signature-based optimal batch size cache |

#### Why LwF Replaced EWC

**EWC Problems:**
- NaN loss explosions after ~50 epochs
- Required careful lambda tuning (100 → 10)
- Fisher Information Matrix computation expensive
- Still caused gradient instability

**LwF Advantages:**
- No parameter-specific penalties
- Uses soft targets from frozen model copy
- Combined with Memory Replay for exemplar-based rehearsal
- More stable training dynamics

#### Training Run Results (Jan 14, 2026)

First full training run with Phase 2 features:

| Teacher | Best Upright | Best Rotation | Duration | Notes |
|---------|-------------|---------------|----------|-------|
| alexnet | - | - | - | NaN on epoch 1 (old architecture) |
| darknet53 | **99.81%** | 98.01% | 10.7m | 6 epochs |
| efficientnet_b0 | 99.72% | 97.76% | 8.4m | 5 epochs |
| inception_v3 | 99.72% | 97.55% | 7.7m | NaN at epoch 5, recovered |
| mobilenet_v2 | **99.91%** | **98.79%** | 8.1m | Best rotation! |
| resnet50 | 99.63% | 98.41% | 8.1m | |
| yolo11n-cls | 99.63% | 98.15% | 8.2m | |
| ensemble_attention | 99.81% | 98.20% | 8.8m | |
| ensemble_concat | **99.91%** | 98.13% | 8.9m | |
| ensemble_cross | 99.53%+ | 97.48%+ | In progress | |

**NaN Regulation in Action:**
- Teacher 4 (inception_v3): LwF strength reduced 1.0 → 0.5 on NaN detection
- Model restored to best state (99.81% accuracy preserved)
- Training continued automatically to next teacher

#### Configuration Defaults (All Enabled)

```python
# Phase 2 defaults in BlackOpsConfig
use_lwf = True                    # LwF enabled
memory_buffer_size = 1000         # 1000 exemplars
use_nan_regulation = True         # Soft NaN handling
auto_deploy = True                # Auto-deploy on success
generate_json_report = True       # Machine-readable
generate_markdown_report = True   # Human-readable
generate_html_report = True       # Interactive
archive_checkpoints = True        # Archive to runs/
enable_experiment_tracking = True # Track all runs

# Disable flags (for ablation studies)
disable_lwf = False
disable_memory_replay = False
disable_nan_regulation = False
disable_mixup = False
disable_cutmix = False
```

#### Test Coverage

- **97 tests passing** (46 Phase 1 + 51 Phase 2)
- All 63 config attributes verified
- No generic naming bugs detected

---

## 10. Phase 8: Production System & Auto-Retraining

### 10.1 Auto-Scheduler System

**Purpose:** Automatically retrain the model when enough user feedback images are collected.

**Thresholds:**
- **Fine-tuning trigger:** 10 images/class OR 150 total images
- **Minimum classes:** At least 3 classes must have images
- **Comprehensive trigger:** 1000 total historical images

**Check Interval:** Every 5 minutes

### 10.2 Retraining Configuration

**Fine-Tuning Parameters:**
- Epochs: 15
- Batch size: 16
- Learning rate: 5e-4
- Weight decay: 0.01
- Gradient clip: 0.5
- Warmup epochs: 2
- Patience: 5 epochs

**Safety Features:**
- EWC (Elastic Weight Consolidation) for preventing catastrophic forgetting
- Collapse detection (>30% accuracy drop → rollback)
- Model backup before each training
- Image archiving after training

### 10.3 Feedback Flow

```
User submits image
        ↓
Model predicts class
        ↓
User provides feedback
        ↓
Image saved to feedback_data/images/
   ├── correct/{class_name}/     ← User confirms prediction
   ├── corrected/{class_name}/   ← User corrects prediction
   └── junk/                     ← User marks as non-pest
        ↓
(After Training)
        ↓
Images archived to model_backups/history/v{N}/
```

### 10.4 Database Schema

**Tables:**
- `users` - User profiles, trust scores, feedback stats
- `submissions` - Image submission records
- `feedback_entries` - User feedback on predictions
- `image_metadata` - Stored image information
- `training_runs` - Training session records
- `training_events` - Per-epoch training logs
- `archived_images` - Images moved after training
- `system_events` - Server/scheduler events
- `scheduler_checks` - Auto-scheduler check history

### 10.5 Current Production State

**Status (Jan 14, 2026):**
- Model Version: v1.0.5
- Total Fine-tunes: 5
- Comprehensive Training: 0
- Auto-Scheduler: ACTIVE (5 min interval)
- Server: Running on port 8000 with Dev Tunnel

---

## 11. Phase 9: Black-Ops Training Phase 2

> **Added: January 14, 2026** - Major upgrade to the ghost training system

See section 9.5 above for full details. Key highlights:

- **LwF + Memory Replay** replaced unstable EWC
- **Soft NaN Regulation** provides graceful degradation instead of hard disables
- **Multi-format reports** (JSON, Markdown, HTML) for comprehensive documentation
- **Auto-deployment** with backup, verification, and rollback capabilities
- **97 tests passing** with full Phase 2 coverage

---

## 12. Phase 10: Phase 3 Implementation (NEW)

> **Added: January 16, 2026** - Region-aware perception and critical bug fixes
> **Updated: January 16, 2026** - Phase 3 fully operational with learning-based region scoring

### Phase 3 Overview

Phase 3 introduces **region-aware perception** infrastructure for spatial analysis capabilities:

**New Features:**
- Feature extraction path from model backbone
- Region proposal grid (7x7 = 49 regions)
- Region feature aggregation with ROI pooling
- **Learning-based region relevance scoring** (uses actual model attention)
- **Multi-label classification head** (detects multiple pests simultaneously)
- Spatial attention map generation

**Configuration Flag:**
```python
# In BlackOpsConfig
enable_phase3: bool = True
phase3_mode: str = "inference"  # "disabled", "inference", "training", "evaluation", "full"
```

### Phase 3 API Response

The API now returns Phase 3 data with every prediction:

```json
{
  "prediction": {
    "class": "army worm",
    "confidence": 0.7738
  },
  "phase3": {
    "executed": true,
    "regions": [
      {"region_id": 0, "bbox": [0,0,32,32], "relevance_score": 0.0072},
      {"region_id": 1, "bbox": [32,0,64,32], "relevance_score": 0.0053},
      ...
    ],
    "top_region_score": 0.048,
    "multi_label": [
      {"label": "army worm", "confidence": 0.9433},
      {"label": "Top borer", "confidence": 0.6208}
    ],
    "processing_time_ms": 81.36
  }
}
```

### Learning-Based Region Scoring

Region relevance scores are computed from **actual model feature activations**:

1. Extract feature map from model's stage4 (7×7 spatial resolution)
2. Compute spatial attention by averaging across channels
3. Map each region's bbox to feature map coordinates
4. Calculate mean activation per region
5. Weight by prediction confidence
6. Normalize scores to sum to 1

**Key Benefit:** Scores **improve as the model is retrained** because better-trained models produce more discriminative feature activations.

**Example Scores (varying by image):**
- Army worm image: top region score ~0.015
- Pink borer image: top region score ~0.048

### Multi-Label Classification

Real sigmoid-based multi-label classification using model logits:

```python
# Multi-label uses actual model logits
class_probabilities = torch.sigmoid(class_logits)
# Returns classes with probability > 0.5
```

**Example Results:**
- Army worm image → army worm (94%), Top borer (62%)
- Pink borer image → Pink borer (95%), root borer (50%)

### Bug Fixes During Phase 3 Implementation

#### Tensor Boolean Evaluation Fix (Critical)

**Issue:** Python `or` operator fails with PyTorch tensors
```python
# BROKEN - RuntimeError: Boolean value of Tensor with more than one value is ambiguous
feature_map = features.stage4_features or features.consolidated_features
```

**Fix:** Proper conditional check
```python
# FIXED
if features.stage4_features is not None:
    feature_map = features.stage4_features
elif features.consolidated_features is not None:
    feature_map = features.consolidated_features
else:
    feature_map = None
```

### Phase 3 API Schemas

New Pydantic schemas added to `inference_server/fastapi_app/schemas.py`:

| Schema | Purpose |
|--------|---------|
| `Phase3Response` | Container for all Phase 3 data |
| `Phase3AttentionInfo` | Spatial attention map data |
| `Phase3RegionInfo` | Region proposal with confidence |
| `Phase3MultiLabelPrediction` | Multi-label prediction results |

The `PredictionResponse` now includes an optional `phase3` field.

### Critical Bug Fixes (Jan 16, 2026)

#### Model Path Mismatch Fix

**Issue:** Server was loading wrong model file
- `run_server.py` loaded: `student_model_12class_proper.pt`
- Ghost trainer deployed to: `student_model_rotation_robust.pt`
- **Result:** Ghost training updates were NOT reflected in server inference

**Fix:** Updated `run_server.py` to use `student_model_rotation_robust.pt`

**Verification:** All 5 training systems now use the same model path:
1. Server startup (`run_server.py`)
2. Ghost trainer (`ghost_config.py`)
3. Comprehensive trainer (`comprehensive_trainer.py`)
4. Retrain API (`routers.py`)
5. Feedback manager (`feedback_manager.py`)

#### App Version System Fix

**Issue:** Version format mismatch causing unnecessary forced updates

| Component | Old Format | New Format |
|-----------|------------|------------|
| Server `timestamp_to_version()` | `YYYY.MMDD.HHMM` | `YYYY.MMDD` |
| Android `build.gradle.kts` | `"1.0.0"` (hardcoded) | `"2026.0116"` |

**Fixes Applied:**
1. Updated `build.gradle.kts` versionName to `"2026.0116"`
2. Changed server version format to match (removed time component)
3. Removed special `"1.0.0"` handling - all versions now compared properly

### Android App Fixes

Kotlin compilation errors fixed after Phase 3 component additions:

| File | Issue | Fix |
|------|-------|-----|
| `CameraScreen.kt` | Wrong AspectRatio imports | Use `CameraPreferences.AspectRatio` |
| `Phase3Visualization.kt` | Missing BorderStroke import | Added import |
| `ResultsScreen.kt` | `isJunk` import error | Removed invalid import |
| `SettingsScreen.kt` | `Pest_Control` icon missing | Changed to `BugReport` |

### Test Results

8 integration tests created and passing:

| Test | Status |
|------|--------|
| Model Path Consistency | ✅ PASS |
| Server Model File Exists | ✅ PASS |
| Model Has 12 Classes (Including Junk) | ✅ PASS |
| Ghost Config Phase 3 Enabled | ✅ PASS |
| Training Utils STUDENT_NUM_CLASSES = 12 | ✅ PASS |
| Phase 3 API Schemas Exist | ✅ PASS |
| Inference Engine Phase 3 Integration | ✅ PASS |
| Live API /classes Endpoint | ✅ PASS |

### Phase 3 System Verification

**Verified Working (Jan 16, 2026):**

| Component | Status | Notes |
|-----------|--------|-------|
| Phase 3 Manager Initialization | ✅ Working | All flags enabled on startup |
| Feature Extraction | ✅ Working | stage4_features populated |
| Region Generation (7x7 Grid) | ✅ Working | 49 regions with bboxes |
| Feature Aggregation | ✅ Working | ROI pooling per region |
| Relevance Scoring | ✅ Working | Varying scores based on attention |
| Multi-Label Classification | ✅ Working | Real sigmoid probabilities |
| Database | ✅ Working | 903 images, 6 users |
| Auto-Retraining | ✅ Working | 6 completed runs (v1.0.1→v1.0.6) |

**Processing Performance:**
- Phase 3 execution: ~80ms additional overhead
- Total inference: ~450ms (including Phase 3)

---

## 13. Issues & Bugs Encountered

### 13.1 Model-Side Issues

| Issue | Description | Resolution |
|-------|-------------|------------|
| aux_classifiers mismatch | Expanding classifier without aux_classifiers | Fixed with proper expansion script |
| NaN loss explosion | EWC lambda too high (100.0) | Reduced to 10.0, added max_ewc_loss cap |
| ensemble_cross training failure | Cross-attention instability | Reduced weight in ensemble |
| Rotation invariance loss | Model lost rotation robustness after fine-tuning | Added rotation validation during training |
| Catastrophic forgetting | Fine-tuning forgot original training | Added EWC and dual-metric tracking |

### 13.2 Conversion-Side Issues

| Issue | Description | Resolution |
|-------|-------------|------------|
| FULLY_CONNECTED v12 | TFLite runtime incompatibility | **Abandoned TFLite, moved to server** |
| KeyError 'split' | DarkNet53 conversion failed | Couldn't fix, excluded model |
| Unicode encoding | Windows console errors | Forced UTF-8 encoding |
| Opset version mismatch | ONNX operations not supported | Used older opset (11) |
| Input size variations | Different models need different sizes | Created metadata JSON |

### 13.3 Server-Side Issues

| Issue | Description | Resolution |
|-------|-------------|------------|
| Memory leaks | ONNX sessions not released | Proper cleanup in finally blocks |
| Concurrent request failures | Asyncio issues | Configured uvicorn properly |
| Database locked | SQLite concurrent access | Thread-local connections |
| Scheduler infinite loop | Continuous retraining triggers | Added cooldown period |
| Model loading race condition | Multiple threads loading | Added thread locks |

### 13.4 Application-Side Issues

| Issue | Description | Resolution |
|-------|-------------|------------|
| Timeout errors | Server inference too slow | Increased timeouts to 90s/120s |
| Emulator networking | Can't reach Dev Tunnel | Special handling for 10.0.2.2 |
| Bitmap hardware config | Processing crashes | Convert to software bitmaps |
| Gallery image loading | Multiple methods needed | 4-method fallback system |
| Location permission | Background tracking denied | Foreground service notification |
| Firebase email verification | Users not receiving emails | Check spam folder notification |
| Large image uploads | Network timeouts | Image compression before upload |

### 12.5 Integration Issues

| Issue | Description | Resolution |
|-------|-------------|------------|
| API key mismatch | App and server keys different | Unified API key |
| Response format changes | Breaking API changes | Version-pinned API |
| Class name inconsistencies | Different formats in app/server | Canonical class names |
| Timestamp format | Server and app using different formats | ISO 8601 standard |

---

## 14. Current State & Architecture

### 14.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTELLI-PEST SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         ┌─────────────────────────────┐   │
│  │  ANDROID APP    │         │    INFERENCE SERVER         │   │
│  │                 │         │                             │   │
│  │  ┌───────────┐  │  HTTPS  │  ┌───────────────────────┐  │   │
│  │  │ Camera    │  │◄───────►│  │ FastAPI + Uvicorn     │  │   │
│  │  │ Gallery   │  │         │  │                       │  │   │
│  │  │ Firebase  │  │         │  │ ┌─────────────────┐   │  │   │
│  │  │ GPS       │  │         │  │ │ PyTorch Model   │   │  │   │
│  │  │ History   │  │         │  │ │ 12-class        │   │  │   │
│  │  └───────────┘  │         │  │ │ 96.25% acc      │   │  │   │
│  │                 │         │  │ └─────────────────┘   │  │   │
│  │  Retrofit2      │         │  │                       │  │   │
│  │  API Client     │         │  │ Auto-Retraining       │  │   │
│  │                 │         │  │ Feedback System       │  │   │
│  └─────────────────┘         │  │ SQLite Database       │  │   │
│                              │  └───────────────────────┘  │   │
│                              │                             │   │
│                              └─────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              VS CODE DEV TUNNELS                         │   │
│  │  https://cnbrr7xn-8000.inc1.devtunnels.ms               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 14.2 Model Specifications

| Specification | Value |
|---------------|-------|
| Architecture | Enhanced CNN with CBAM, FPN |
| Parameters | 12.2M |
| Size (PyTorch) | 46.85 MB |
| Input | 256×256×3 RGB |
| Output | 12 classes |
| Accuracy | 96.25% (11-class validation) |

### 14.3 Class Definitions (12 Classes)

| Index | Class | Description |
|-------|-------|-------------|
| 0 | Healthy | No pest damage |
| 1 | Internode borer | Damage to internodes |
| 2 | Pink borer | Pink-colored borer damage |
| 3 | Rat damage | Gnaw marks from rats |
| 4 | Stalk borer | Damage to sugarcane stalk |
| 5 | Top borer | Damage at top of plant |
| 6 | army worm | Leaf damage from army worms |
| 7 | mealy bug | White insect infestation |
| 8 | porcupine damage | Large animal damage |
| 9 | root borer | Damage to roots |
| 10 | termite | Termite infestation |
| 11 | junk | Non-pest images (filter class) |

### 14.4 File Locations

**Models:**
- Current: `D:\KnowledgeDistillation\student_model_rotation_robust.pt` (ghost trainer target)
- Backup 12-class: `D:\KnowledgeDistillation\student_model_12class_proper.pt`
- Original 11-class: `D:\KnowledgeDistillation\student_model_final.pth`
- Backups: `D:\Intelli_PEST-Backend\model_backups\`

**Server:**
- Main script: `D:\Intelli_PEST-Backend\run_server.py`
- Inference server: `D:\Intelli_PEST-Backend\inference_server\`
- Feedback data: `D:\Intelli_PEST-Backend\feedback_data\`

**App:**
- Project: `D:\App\Intelli_PEST\`
- APK: `D:\App\Intelli_PEST\intelli-pest-release-2026-01-12.apk`

---

## 15. Lessons Learned

### 15.1 Technical Lessons

1. **Framework compatibility is critical** - Always test target runtime versions before committing to a deployment strategy.

2. **On-device inference has hidden costs** - Maintaining compatibility across Android versions is a significant burden.

3. **Server-side inference simplifies updates** - Model improvements don't require app store submissions.

4. **Knowledge distillation works** - Multiple teachers can produce a student that outperforms any single teacher.

5. **Rotation augmentation is essential** - Real-world images come in all orientations.

6. **EWC needs careful tuning** - High lambda values cause NaN explosions.

7. **Always backup before retraining** - Model corruption can happen during training.

### 15.2 Process Lessons

1. **Pivot when necessary** - The TFLite compatibility issues warranted a complete architecture change.

2. **Document everything** - This journey documentation helps future troubleshooting.

3. **Test on real devices** - Emulators don't catch all compatibility issues.

4. **Plan for model evolution** - The 12-class expansion was necessary but required careful implementation.

5. **User feedback is valuable** - The auto-retraining system leverages real-world data.

### 15.3 What We Would Do Differently

1. **Start with server-side** - Would have saved months of TFLite debugging.

2. **Plan for class expansion** - Original architecture should have anticipated junk class.

3. **Implement CI/CD earlier** - Automated testing would have caught issues faster.

4. **Use standard class names** - Inconsistent naming (army worm vs Armyworm) caused confusion.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Jan 14, 2026 | GitHub Copilot | Initial comprehensive documentation |
| 1.1 | Jan 14, 2026 | GitHub Copilot | Added Phase 2 Black-Ops Training (LwF, NaN Regulation, Reports, Auto-Deploy) |
| 1.2 | Jan 16, 2026 | GitHub Copilot | Added Phase 3 Implementation, fixed model path mismatch, app version system |
| 1.3 | Jan 16, 2026 | GitHub Copilot | Phase 3 fully operational: learning-based region scoring, multi-label classification, tensor evaluation fix |

---

**End of Document**
