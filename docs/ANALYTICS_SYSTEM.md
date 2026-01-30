# Performance Correction Tracking System

A comprehensive analytics system for tracking model predictions, user corrections, and performance metrics over time.

## 📦 Components

```
src/analytics/
├── __init__.py                 # Module exports
├── correction_tracker.py       # Core logging (JSON + SQLite)
├── performance_analytics.py    # Metrics computation engine
├── performance_dashboard.py    # Charts, reports, exports
├── integration.py             # Server integration helpers
└── cli.py                     # Command-line interface
```

## 🚀 Quick Start

### Server Integration

Add to your server startup:

```python
from src.analytics import init_analytics

# Initialize on server startup
init_analytics(model_version="v1.0.6")
```

### Logging Predictions

After each inference call:

```python
from src.analytics.integration import log_prediction

log_prediction(
    prediction_id=feedback_id,
    predicted_class="army worm",
    confidence=0.87,
    image_hash=image_hash,
    user_id=user_id
)
```

### Logging Corrections

When feedback is submitted:

```python
from src.analytics.integration import log_correction

log_correction(
    prediction_id=feedback_id,
    predicted_class="army worm",
    actual_class="Stalk borer",
    is_correct=False,
    confidence=0.87
)
```

## 📊 CLI Commands

Generate reports from the command line:

```bash
# Quick summary to console
python -m src.analytics.cli summary

# Full report with charts, Markdown, and CSV
python -m src.analytics.cli report

# Charts only
python -m src.analytics.cli charts

# CSV exports only
python -m src.analytics.cli csv

# Interactive dashboard
python -m src.analytics.cli dashboard
```

## 📁 Storage

Data is stored in dual format for reliability and flexibility:

### JSON Storage
- **Location**: `feedback_data/analytics/corrections_log.json`
- **Format**: Human-readable JSON
- **Features**: Daily auto-save, version control friendly

### SQLite Database
- **Location**: `feedback_data/analytics/corrections.db`
- **Tables**:
  - `corrections`: Individual prediction records
  - `daily_summary`: Aggregated daily metrics
  - `class_metrics`: Per-class performance
  - `model_versions`: Model version history

## 📈 Generated Reports

### Charts (`feedback_data/analytics/charts/`)
| Chart | Description |
|-------|-------------|
| `accuracy_trend_daily.png` | Daily accuracy over time |
| `accuracy_trend_weekly.png` | Weekly accuracy trends |
| `confusion_matrix.png` | Prediction vs actual heatmap |
| `class_accuracy.png` | Per-class accuracy bars |
| `class_precision.png` | Per-class precision bars |
| `class_recall.png` | Per-class recall bars |
| `correction_frequency.png` | Most common misclassifications |
| `model_comparison.png` | Version-to-version comparison |

### Markdown Reports (`feedback_data/analytics/reports/`)
- Comprehensive performance report
- Executive summary
- Per-class metrics table
- Trend analysis
- Improvement recommendations

### CSV Exports (`feedback_data/analytics/exports/`)
| File | Content |
|------|---------|
| `overall_metrics.csv` | Global accuracy, precision metrics |
| `class_metrics.csv` | Per-class performance data |
| `time_series_daily.csv` | Daily metrics history |
| `time_series_weekly.csv` | Weekly metrics history |
| `confusion_matrix.csv` | Full confusion matrix |
| `corrections.csv` | All recorded corrections |

## 📊 Metrics Tracked

### Overall Metrics
- Total predictions
- Correct/incorrect counts
- Overall accuracy
- Average confidence
- Confidence calibration

### Per-Class Metrics
- Class-wise accuracy
- Precision (TP / (TP + FP))
- Recall (TP / (TP + FN))
- F1 Score
- Average confidence per class

### Trend Analysis
- Week-over-week improvement
- Class improvement trends
- Model version comparisons

### Attention Analysis
- Attention map correlation with correctness
- Region relevance tracking (when enabled)

## 🔧 Configuration

The analytics system auto-configures, but can be customized:

```python
from src.analytics import CorrectionTracker

tracker = CorrectionTracker(
    data_dir="custom/analytics/path",
    auto_save_interval=60,  # Save every 60 seconds
    enable_sqlite=True,
    enable_json=True
)
```

## 📱 Dashboard

The HTML dashboard provides an interactive overview:

```bash
python -m src.analytics.cli dashboard
```

Features:
- Real-time metrics cards
- Embedded charts
- Per-class performance table
- Trend indicators
- Color-coded status (green/yellow/red)

## 🔄 Workflow

1. **Prediction Logged**: Every inference call logs prediction details
2. **Feedback Received**: User confirms or corrects prediction
3. **Correction Logged**: System records the correction
4. **Metrics Updated**: Daily summaries and class metrics update
5. **Reports Generated**: On-demand via CLI or dashboard

## 📝 Example Session

```bash
# Server starts, initializes analytics
[INFO] Analytics initialized with model version: v1.0.6

# Predictions logged automatically
[DEBUG] Logged prediction: pred_001 -> army worm (87.2%)
[DEBUG] Logged prediction: pred_002 -> Stalk borer (92.1%)

# User submits feedback
[INFO] Logged correction: pred_001 -> army worm → Stalk borer

# Generate report
$ python -m src.analytics.cli report

📊 Generating full report...
   Output: feedback_data/analytics/report_20260115_143022

✅ Report generated successfully!
   📈 Charts: 8 files
   📄 Reports: 1 files
   📋 Exports: 6 files

📄 Main report: feedback_data/analytics/reports/performance_report_2026-01-15.md
```

## 🔍 Troubleshooting

### No Data Appearing
- Ensure `init_analytics()` is called on server startup
- Verify `log_prediction()` is called after each inference
- Check that the data directory is writable

### Charts Not Generating
- Install matplotlib: `pip install matplotlib`
- Install numpy: `pip install numpy`

### SQLite Errors
- Ensure SQLite3 is available (built into Python)
- Check disk space for database file

## 📚 API Reference

### CorrectionTracker
```python
tracker.log_prediction(
    prediction_id: str,
    predicted_class: str,
    predicted_class_id: int,
    confidence: float,
    image_hash: str = None,
    user_id: str = None,
    attention_map: Any = None,
    metadata: Dict = None
) -> str

tracker.log_correction(
    prediction_id: str,
    predicted_class: str,
    actual_class: str,
    is_correct: bool,
    confidence: float = None,
    attention_map: Any = None,
    metadata: Dict = None
) -> str
```

### PerformanceAnalytics
```python
analytics.get_overall_metrics() -> Dict
analytics.get_class_metrics() -> Dict[str, ClassMetrics]
analytics.get_confusion_matrix() -> ConfusionData
analytics.get_time_series(granularity="daily") -> List[TimeSeriesMetrics]
analytics.get_improvement_trends() -> Dict
analytics.get_model_comparison() -> Dict
```

### PerformanceDashboard
```python
dashboard.generate_all_charts() -> List[str]
dashboard.generate_markdown_report() -> str
dashboard.export_all_csv() -> List[str]
dashboard.generate_full_report() -> Dict
```
