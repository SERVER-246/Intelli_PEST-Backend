"""
PERFORMANCE ANALYTICS ENGINE
============================
Computes accuracy, precision, recall, trends, and confusion matrices.

Features:
- Time-series analysis (daily/weekly/monthly trends)
- Per-class metrics with precision/recall/F1
- Confusion matrix analysis
- Improvement tracking after corrections
- Attention heatmap correlation analysis

Usage:
    from analytics import PerformanceAnalytics
    
    analytics = PerformanceAnalytics()
    
    # Get overall metrics
    metrics = analytics.get_overall_metrics()
    
    # Get per-class breakdown
    class_metrics = analytics.get_class_metrics()
    
    # Get confusion matrix
    confusion = analytics.get_confusion_matrix(start_date="2026-01-01")
    
    # Get improvement trends
    trends = analytics.get_improvement_trends()
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    class_name: str
    total_predictions: int = 0
    correct: int = 0
    incorrect: int = 0
    
    # As predicted class
    true_positives: int = 0   # Predicted this class, was correct
    false_positives: int = 0  # Predicted this class, was wrong
    
    # As actual class
    false_negatives: int = 0  # Should have been this class, predicted something else
    
    avg_confidence: float = 0.0
    avg_confidence_when_correct: float = 0.0
    avg_confidence_when_wrong: float = 0.0
    
    @property
    def accuracy(self) -> float:
        """Class-specific accuracy."""
        total = self.correct + self.incorrect
        return self.correct / total if total > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with computed metrics."""
        d = asdict(self)
        d['accuracy'] = self.accuracy
        d['precision'] = self.precision
        d['recall'] = self.recall
        d['f1_score'] = self.f1_score
        return d


@dataclass
class TimeSeriesMetrics:
    """Metrics for a time period."""
    date: str
    total_predictions: int = 0
    correct: int = 0
    incorrect: int = 0
    pending: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    model_version: str = ""
    
    # Per-class breakdown for this period
    class_breakdown: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class ConfusionData:
    """Confusion matrix data."""
    classes: List[str]
    matrix: List[List[int]]  # matrix[true_idx][pred_idx] = count
    
    def get_confused_pairs(self, min_count: int = 2) -> List[Tuple[str, str, int]]:
        """Get pairs of commonly confused classes."""
        pairs = []
        for i, true_class in enumerate(self.classes):
            for j, pred_class in enumerate(self.classes):
                if i != j and self.matrix[i][j] >= min_count:
                    pairs.append((true_class, pred_class, self.matrix[i][j]))
        return sorted(pairs, key=lambda x: -x[2])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "classes": self.classes,
            "matrix": self.matrix,
            "confused_pairs": self.get_confused_pairs()
        }


@dataclass  
class AttentionCorrelation:
    """Correlation between attention patterns and correctness."""
    class_name: str
    avg_entropy_correct: float = 0.0
    avg_entropy_incorrect: float = 0.0
    avg_focus_correct: float = 0.0
    avg_focus_incorrect: float = 0.0
    
    @property
    def entropy_difference(self) -> float:
        """Difference in attention entropy (correct - incorrect)."""
        return self.avg_entropy_correct - self.avg_entropy_incorrect
    
    @property
    def focus_difference(self) -> float:
        """Difference in attention focus (correct - incorrect)."""
        return self.avg_focus_correct - self.avg_focus_incorrect


class PerformanceAnalytics:
    """
    Analytics engine for computing performance metrics.
    
    Reads from the SQLite database populated by CorrectionTracker.
    """
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize analytics engine.
        
        Args:
            data_dir: Directory containing corrections.db
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "feedback_data" / "analytics"
        
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "corrections.db"
        
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def get_overall_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get overall performance metrics.
        
        Args:
            start_date: Filter from date (YYYY-MM-DD)
            end_date: Filter to date (YYYY-MM-DD)
            model_version: Filter by model version
            
        Returns:
            Dictionary with overall metrics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Build query
        query = 'SELECT * FROM corrections WHERE was_correct IS NOT NULL'
        params = []
        
        if start_date:
            query += ' AND prediction_timestamp >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND prediction_timestamp <= ?'
            params.append(end_date + 'T23:59:59')
        if model_version:
            query += ' AND model_version = ?'
            params.append(model_version)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        # Compute metrics
        total = len(rows)
        correct = sum(1 for r in rows if r[12] == 1)  # was_correct column
        incorrect = total - correct
        
        # Confidence analysis
        confidences = [r[4] for r in rows]  # predicted_confidence column
        correct_confidences = [r[4] for r in rows if r[12] == 1]
        incorrect_confidences = [r[4] for r in rows if r[12] == 0]
        
        return {
            "period": {
                "start": start_date or "all time",
                "end": end_date or "now"
            },
            "model_version": model_version or "all versions",
            "total_predictions": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": correct / total if total > 0 else 0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "avg_confidence_correct": sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0,
            "avg_confidence_incorrect": sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0,
            "confidence_calibration": self._compute_calibration(rows)
        }
    
    def _compute_calibration(self, rows: List) -> Dict[str, float]:
        """Compute confidence calibration (expected calibration error)."""
        # Bin predictions by confidence
        bins = defaultdict(list)
        for row in rows:
            conf = row[4]  # predicted_confidence
            correct = row[12] == 1  # was_correct
            bin_idx = min(int(conf * 10), 9)  # 0-9 for 0-100%
            bins[bin_idx].append((conf, correct))
        
        calibration = {}
        for bin_idx, values in sorted(bins.items()):
            if values:
                avg_conf = sum(v[0] for v in values) / len(values)
                actual_acc = sum(1 for v in values if v[1]) / len(values)
                calibration[f"{bin_idx*10}-{(bin_idx+1)*10}%"] = {
                    "avg_confidence": avg_conf,
                    "actual_accuracy": actual_acc,
                    "gap": abs(avg_conf - actual_acc),
                    "count": len(values)
                }
        
        return calibration
    
    def get_class_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> Dict[str, ClassMetrics]:
        """
        Get per-class performance metrics.
        
        Returns:
            Dictionary mapping class names to ClassMetrics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Build query
        query = 'SELECT * FROM corrections WHERE was_correct IS NOT NULL'
        params = []
        
        if start_date:
            query += ' AND prediction_timestamp >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND prediction_timestamp <= ?'
            params.append(end_date + 'T23:59:59')
        if model_version:
            query += ' AND model_version = ?'
            params.append(model_version)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        # Compute per-class metrics
        class_data = defaultdict(lambda: ClassMetrics(class_name=""))
        
        for row in rows:
            pred_class = row['predicted_class']
            actual_class = row['actual_class']
            was_correct = row['was_correct'] == 1
            confidence = row['predicted_confidence']
            
            # Initialize class names
            if class_data[pred_class].class_name == "":
                class_data[pred_class].class_name = pred_class
            if actual_class and class_data[actual_class].class_name == "":
                class_data[actual_class].class_name = actual_class
            
            # Update predicted class metrics
            class_data[pred_class].total_predictions += 1
            
            if was_correct:
                class_data[pred_class].correct += 1
                class_data[pred_class].true_positives += 1
                class_data[pred_class].avg_confidence_when_correct = (
                    (class_data[pred_class].avg_confidence_when_correct * 
                     (class_data[pred_class].correct - 1) + confidence) / 
                    class_data[pred_class].correct
                )
            else:
                class_data[pred_class].incorrect += 1
                class_data[pred_class].false_positives += 1
                class_data[pred_class].avg_confidence_when_wrong = (
                    (class_data[pred_class].avg_confidence_when_wrong * 
                     (class_data[pred_class].incorrect - 1) + confidence) / 
                    class_data[pred_class].incorrect
                )
                
                # Update actual class false negatives
                if actual_class:
                    class_data[actual_class].false_negatives += 1
            
            # Update average confidence
            n = class_data[pred_class].total_predictions
            class_data[pred_class].avg_confidence = (
                (class_data[pred_class].avg_confidence * (n - 1) + confidence) / n
            )
        
        return dict(class_data)
    
    def get_confusion_matrix(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> ConfusionData:
        """
        Get confusion matrix.
        
        Returns:
            ConfusionData with matrix and class list
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get all unique classes
        cursor.execute('''
            SELECT DISTINCT predicted_class FROM corrections WHERE was_correct IS NOT NULL
            UNION
            SELECT DISTINCT actual_class FROM corrections WHERE actual_class IS NOT NULL
        ''')
        classes = sorted([r[0] for r in cursor.fetchall() if r[0]])
        
        # Build query
        query = '''
            SELECT predicted_class, actual_class, COUNT(*) as cnt
            FROM corrections 
            WHERE was_correct IS NOT NULL AND actual_class IS NOT NULL
        '''
        params = []
        
        if start_date:
            query += ' AND prediction_timestamp >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND prediction_timestamp <= ?'
            params.append(end_date + 'T23:59:59')
        if model_version:
            query += ' AND model_version = ?'
            params.append(model_version)
        
        query += ' GROUP BY predicted_class, actual_class'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        # Build matrix
        class_to_idx = {c: i for i, c in enumerate(classes)}
        matrix = [[0] * len(classes) for _ in range(len(classes))]
        
        for pred_class, actual_class, count in rows:
            if pred_class in class_to_idx and actual_class in class_to_idx:
                pred_idx = class_to_idx[pred_class]
                actual_idx = class_to_idx[actual_class]
                matrix[actual_idx][pred_idx] = count
        
        return ConfusionData(classes=classes, matrix=matrix)
    
    def get_time_series(
        self,
        granularity: str = "daily",  # daily, weekly, monthly
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> List[TimeSeriesMetrics]:
        """
        Get time series metrics.
        
        Args:
            granularity: "daily", "weekly", or "monthly"
            
        Returns:
            List of TimeSeriesMetrics ordered by date
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Date grouping based on granularity
        if granularity == "weekly":
            date_expr = "strftime('%Y-W%W', prediction_timestamp)"
        elif granularity == "monthly":
            date_expr = "strftime('%Y-%m', prediction_timestamp)"
        else:  # daily
            date_expr = "date(prediction_timestamp)"
        
        query = f'''
            SELECT 
                {date_expr} as period,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as incorrect,
                SUM(CASE WHEN was_correct IS NULL THEN 1 ELSE 0 END) as pending,
                AVG(predicted_confidence) as avg_conf,
                model_version
            FROM corrections
            WHERE 1=1
        '''
        params = []
        
        if start_date:
            query += ' AND prediction_timestamp >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND prediction_timestamp <= ?'
            params.append(end_date + 'T23:59:59')
        if model_version:
            query += ' AND model_version = ?'
            params.append(model_version)
        
        query += f' GROUP BY {date_expr} ORDER BY period'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        results = []
        for row in rows:
            period, total, correct, incorrect, pending, avg_conf, version = row
            validated = correct + incorrect
            accuracy = correct / validated if validated > 0 else 0
            
            results.append(TimeSeriesMetrics(
                date=period,
                total_predictions=total,
                correct=correct,
                incorrect=incorrect,
                pending=pending,
                accuracy=accuracy,
                avg_confidence=avg_conf or 0,
                model_version=version or ""
            ))
        
        return results
    
    def get_improvement_trends(
        self,
        compare_periods: int = 4,
        granularity: str = "weekly"
    ) -> Dict[str, Any]:
        """
        Analyze improvement trends over time.
        
        Returns:
            Dictionary with trend analysis
        """
        time_series = self.get_time_series(granularity=granularity)
        
        if len(time_series) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Compare recent vs earlier periods
        recent = time_series[-compare_periods:] if len(time_series) >= compare_periods else time_series
        earlier = time_series[:compare_periods] if len(time_series) >= compare_periods * 2 else time_series[:len(time_series)//2]
        
        recent_acc = sum(t.accuracy for t in recent) / len(recent) if recent else 0
        earlier_acc = sum(t.accuracy for t in earlier) / len(earlier) if earlier else 0
        
        improvement = recent_acc - earlier_acc
        
        # Per-class trends
        class_trends = self._compute_class_trends(granularity)
        
        return {
            "overall_trend": {
                "earlier_accuracy": earlier_acc,
                "recent_accuracy": recent_acc,
                "improvement": improvement,
                "improvement_pct": improvement * 100,
                "trend": "improving" if improvement > 0.01 else ("declining" if improvement < -0.01 else "stable")
            },
            "periods_analyzed": len(time_series),
            "class_trends": class_trends,
            "top_improving": sorted(
                [(k, v['improvement']) for k, v in class_trends.items()],
                key=lambda x: -x[1]
            )[:5],
            "needs_attention": sorted(
                [(k, v['improvement']) for k, v in class_trends.items()],
                key=lambda x: x[1]
            )[:5]
        }
    
    def _compute_class_trends(self, granularity: str = "weekly") -> Dict[str, Dict]:
        """Compute per-class improvement trends."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get class metrics by period
        if granularity == "weekly":
            date_expr = "strftime('%Y-W%W', prediction_timestamp)"
        elif granularity == "monthly":
            date_expr = "strftime('%Y-%m', prediction_timestamp)"
        else:
            date_expr = "date(prediction_timestamp)"
        
        cursor.execute(f'''
            SELECT 
                {date_expr} as period,
                predicted_class,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as incorrect
            FROM corrections
            WHERE was_correct IS NOT NULL
            GROUP BY period, predicted_class
            ORDER BY period
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        # Organize by class and period
        class_periods = defaultdict(list)
        for period, cls, correct, incorrect in rows:
            total = correct + incorrect
            acc = correct / total if total > 0 else 0
            class_periods[cls].append((period, acc, total))
        
        # Compute trends
        trends = {}
        for cls, periods in class_periods.items():
            if len(periods) < 2:
                trends[cls] = {"improvement": 0, "data_points": len(periods)}
                continue
            
            # Compare first half vs second half
            mid = len(periods) // 2
            earlier = periods[:mid]
            recent = periods[mid:]
            
            earlier_acc = sum(p[1] for p in earlier) / len(earlier)
            recent_acc = sum(p[1] for p in recent) / len(recent)
            
            trends[cls] = {
                "earlier_accuracy": earlier_acc,
                "recent_accuracy": recent_acc,
                "improvement": recent_acc - earlier_acc,
                "data_points": len(periods),
                "total_samples": sum(p[2] for p in periods)
            }
        
        return trends
    
    def get_attention_analysis(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, AttentionCorrelation]:
        """
        Analyze attention patterns and their correlation with correctness.
        
        Returns:
            Per-class attention correlation analysis
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = '''
            SELECT predicted_class, was_correct, attention_info, predicted_confidence
            FROM corrections
            WHERE was_correct IS NOT NULL AND attention_info IS NOT NULL
        '''
        params = []
        
        if start_date:
            query += ' AND prediction_timestamp >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND prediction_timestamp <= ?'
            params.append(end_date + 'T23:59:59')
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Aggregate by class
        class_data = defaultdict(lambda: {
            "entropy_correct": [], "entropy_incorrect": [],
            "focus_correct": [], "focus_incorrect": []
        })
        
        for pred_class, was_correct, attention_json, confidence in rows:
            try:
                attention = json.loads(attention_json) if attention_json else {}
                entropy = attention.get('entropy', 0)
                focus = attention.get('focus_area_pct', 0)
                
                if was_correct:
                    class_data[pred_class]["entropy_correct"].append(entropy)
                    class_data[pred_class]["focus_correct"].append(focus)
                else:
                    class_data[pred_class]["entropy_incorrect"].append(entropy)
                    class_data[pred_class]["focus_incorrect"].append(focus)
            except:
                continue
        
        # Compute correlations
        results = {}
        for cls, data in class_data.items():
            results[cls] = AttentionCorrelation(
                class_name=cls,
                avg_entropy_correct=sum(data["entropy_correct"]) / len(data["entropy_correct"]) if data["entropy_correct"] else 0,
                avg_entropy_incorrect=sum(data["entropy_incorrect"]) / len(data["entropy_incorrect"]) if data["entropy_incorrect"] else 0,
                avg_focus_correct=sum(data["focus_correct"]) / len(data["focus_correct"]) if data["focus_correct"] else 0,
                avg_focus_incorrect=sum(data["focus_incorrect"]) / len(data["focus_incorrect"]) if data["focus_incorrect"] else 0
            )
        
        return results
    
    def get_correction_frequency(
        self,
        top_n: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get most frequently corrected class pairs.
        
        Returns:
            List of (predicted, actual, count) tuples for misclassifications
        """
        confusion = self.get_confusion_matrix(start_date=start_date, end_date=end_date)
        pairs = confusion.get_confused_pairs(min_count=1)
        
        return [
            {
                "predicted": pred,
                "actual": actual, 
                "count": count,
                "description": f"Predicted '{pred}' but was actually '{actual}'"
            }
            for pred, actual, count in pairs[:top_n]
        ]
    
    def get_model_comparison(self) -> Dict[str, Dict]:
        """
        Compare performance across model versions.
        
        Returns:
            Dictionary mapping version to metrics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                model_version,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as incorrect,
                AVG(predicted_confidence) as avg_conf,
                MIN(prediction_timestamp) as first_seen,
                MAX(prediction_timestamp) as last_seen
            FROM corrections
            WHERE was_correct IS NOT NULL
            GROUP BY model_version
            ORDER BY first_seen
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        results = {}
        for version, total, correct, incorrect, avg_conf, first, last in rows:
            validated = correct + incorrect
            results[version] = {
                "total_predictions": total,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": correct / validated if validated > 0 else 0,
                "avg_confidence": avg_conf or 0,
                "first_seen": first,
                "last_seen": last,
                "active_days": (
                    datetime.fromisoformat(last) - datetime.fromisoformat(first)
                ).days + 1 if first and last else 0
            }
        
        return results
