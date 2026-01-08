# Intelli-PEST Backend System Architecture

## Overview

The Intelli-PEST backend is an intelligent pest detection system with automatic model retraining capabilities. This document explains how all components work together.

---

## System Components

### 1. Inference Server (`run_server.py`)
The main entry point that starts all services:

```
Server Startup Flow:
1. Initialize SQLite Database (feedback_data/intellipest.db)
2. Initialize User Tracker (feedback_data/users/)
3. Initialize Data Collector (feedback_data/)
4. Initialize Feedback Manager
5. Initialize Retraining Manager
6. Start Auto-Scheduler (checks every 5 minutes)
7. Start FastAPI server on port 8000
```

### 2. Database System (`feedback/database.py`)

SQLite database with these tables:

| Table | Purpose |
|-------|---------|
| `users` | User profiles, trust scores, feedback stats |
| `submissions` | Image submission records |
| `feedback_entries` | User feedback on predictions |
| `image_metadata` | Stored image information |
| `audit_log` | Admin actions and changes |
| `training_runs` | Training session records |
| `training_events` | Per-epoch/batch training logs |
| `archived_images` | Images moved after training |
| `system_events` | Server/scheduler events |
| `scheduler_checks` | Auto-scheduler check history |

### 3. User Tracking (`feedback/user_tracker.py`)

**FLAGGING DISABLED** - All users are treated as trusted experts.

```python
FLAGGING_ENABLED = False  # Users are trusted experts
```

- Tracks user submissions and corrections
- Records location data (GPS)
- Maintains trust scores (all users start at 100)

### 4. Retraining Manager (`training/retrain_manager.py`)

Handles **regular fine-tuning** when thresholds are met:

**Thresholds:**
- 10 images per class OR 150 total images
- Minimum 3 classes must have images (prevents bias)

**Features:**
- EWC (Elastic Weight Consolidation) - prevents catastrophic forgetting
- Gradient clipping (max_norm=1.0)
- Dual-metric tracking (upright + rotation accuracy)
- Collapse detection (>30% accuracy drop → rollback)
- Image archiving after training

### 5. Comprehensive Trainer (`training/comprehensive_trainer.py`)

Handles **full 360° rotation-robust training** at 1000+ images:

**Trigger:** 1000 total historical feedback images

**Features:**
- Full 360° rotation augmentation (24 angles)
- 50 epochs (vs 15 for fine-tuning)
- Checkpoint recovery (survives power outages)
- Uses ALL data sources:
  - Current feedback images
  - Archived feedback from previous trainings
  - Original datasets

---

## Auto-Scheduler Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SERVER STARTS                             │
│                         ↓                                    │
│            Initialize Database                               │
│                         ↓                                    │
│         Start Auto-Scheduler (5 min interval)               │
└─────────────────────────────────────────────────────────────┘
                          ↓
              ┌───────────────────────┐
              │  SCHEDULER CHECK      │ (every 5 minutes)
              │  ↓                    │
              │  Count feedback images│
              │  ↓                    │
              │  Log to database      │
              └───────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
  Total < 1000                        Total ≥ 1000
        ↓                                   ↓
  Check Fine-Tune Thresholds     Check Comprehensive Thresholds
        ↓                                   ↓
  10/class OR 150 total?         Already completed?
  Min 3 classes have images?            ↓
        ↓                         No → Trigger Comprehensive
        ↓                               ↓
  Yes → Trigger Fine-Tuning      50 epochs, all data
        ↓                               ↓
  15 epochs, EWC                        ↓
        ↓                               ↓
        └───────────┬───────────────────┘
                    ↓
           Archive Images After Training
           (Move to model_backups/history/v{N}/)
                    ↓
           Log training completion to DB
                    ↓
           Wait for next scheduler check
```

---

## Archive Paths (NO CONFLICT)

| Training Type | Archive Path |
|---------------|--------------|
| Fine-Tuning (v1, v2, ...) | `model_backups/history/v1/`, `v2/`, etc. |
| Comprehensive (v1, v2, ...) | `model_backups/history/v1_comprehensive/`, `v2_comprehensive/`, etc. |

**Key Point:** The `_comprehensive` suffix ensures archives never overlap!

---

## Training Thresholds Comparison

| Metric | Fine-Tuning | Comprehensive |
|--------|-------------|---------------|
| Trigger | 10/class OR 150 total | 1000 total historical |
| Epochs | 15 | 50 |
| Data Sources | Current feedback only | All sources |
| EWC Lambda | 100.0 | 100.0 |
| Patience | 5 epochs | 8 epochs |
| Archive Path | `v{N}/` | `v{N}_comprehensive/` |

---

## Feedback Image Flow

```
User submits image
        ↓
Model predicts class
        ↓
User provides feedback
        ↓
┌───────────────────────────────────────┐
│ feedback_data/images/                  │
│   ├── correct/{class_name}/           │ ← User confirms prediction
│   ├── corrected/{class_name}/         │ ← User corrects prediction
│   └── junk/                           │ ← User marks as non-pest
└───────────────────────────────────────┘
        ↓
(After Training)
        ↓
┌───────────────────────────────────────┐
│ model_backups/history/                 │
│   ├── v1/                             │ ← Fine-tuning run 1
│   │   ├── correct/                    │
│   │   ├── corrected/                  │
│   │   └── junk/                       │
│   ├── v2/                             │ ← Fine-tuning run 2
│   ├── v1_comprehensive/               │ ← Comprehensive run 1
│   └── ...                             │
└───────────────────────────────────────┘
```

---

## Database Logging

Every action is logged for comprehensive tracking:

### System Events
- Server start/stop
- Scheduler start/stop
- Errors and warnings

### Scheduler Checks
- Timestamp of each check
- Image counts at check time
- Whether thresholds were met
- Whether training was triggered

### Training Runs
- Run ID, type, status
- Start/end times
- Images used per class
- Accuracy metrics (upright + rotation)
- Early stopping, collapse detection
- Duration in seconds

### Training Events (per-epoch)
- Epoch number
- Loss values
- Accuracy metrics
- Checkpoint saves

### Archived Images
- Original path
- Archive path
- Training run that used the image
- Class name

---

## Generating Reports

Run the report generator to export all data to Excel:

```bash
cd d:\Intelli_PEST-Backend
python scripts/generate_report.py
```

Output: `reports/intellipest_report_YYYYMMDD_HHMMSS.xlsx`

Sheets included:
1. Summary Dashboard - Key statistics
2. Users - All user records
3. Training Runs - All training sessions
4. Scheduler Checks - Auto-scheduler history
5. System Events - Server/scheduler logs
6. Archived Images - Post-training archives
7. Feedback Entries - User feedback records
8. Audit Log - Admin actions

---

## Safety Features

### 1. Archive Fail-Safe
Images are moved (not deleted) after training. They remain in history for:
- Future comprehensive training
- Audit trail
- Recovery if needed

### 2. Model Backup
Before any training:
- Current model is backed up with timestamp
- Stored in `model_backups/`

### 3. Collapse Detection
If accuracy drops >30%:
- Training stops immediately
- Best model is restored
- Logged to database

### 4. EWC (Elastic Weight Consolidation)
Prevents catastrophic forgetting:
- Preserves important weights from original training
- Allows learning new patterns without losing old ones

### 5. Checkpoint Recovery
Comprehensive training saves checkpoints:
- Survives power outages
- Can resume from last epoch
- State saved every 2 epochs

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Submit image for prediction |
| `/feedback` | POST | Submit user feedback |
| `/training/status` | GET | Get training status |
| `/training/trigger` | POST | Manually trigger training |
| `/admin/users` | GET | List all users |
| `/admin/stats` | GET | Get system statistics |

---

## Troubleshooting

### Training not triggering?
1. Check scheduler is running: Look for "Auto-scheduler started" in logs
2. Verify thresholds: Need 10/class OR 150 total + min 3 classes
3. Check database: `SELECT * FROM scheduler_checks ORDER BY check_timestamp DESC LIMIT 10`

### Images not being archived?
1. Check training completed successfully
2. Look for "Archived X images" in logs
3. Verify archive path exists: `model_backups/history/`

### Database not logging?
1. Ensure database is initialized before scheduler starts
2. Check `feedback_data/intellipest.db` exists
3. Look for "Database initialized" in startup logs

---

## Quick Reference

```
Feedback Images:     feedback_data/images/
Archive Location:    model_backups/history/
Model Backups:       model_backups/
Database:            feedback_data/intellipest.db
Reports:             reports/

Fine-Tune Trigger:   10/class OR 150 total (min 3 classes)
Comprehensive:       1000 total historical images
Scheduler Interval:  5 minutes
```
