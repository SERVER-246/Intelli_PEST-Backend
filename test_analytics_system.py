#!/usr/bin/env python3
"""
Test Analytics System
=====================
Verifies all components of the Performance Correction Tracking System.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


def test_correction_tracker():
    """Test the CorrectionTracker component."""
    print("\n" + "=" * 60)
    print("TEST: CorrectionTracker")
    print("=" * 60)
    
    from src.analytics.correction_tracker import CorrectionTracker
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="analytics_test_")
    
    try:
        tracker = CorrectionTracker(data_dir=temp_dir)
        tracker.set_model_version("v1.0.6-test")
        
        # Log some predictions
        classes = ["army worm", "Stalk borer", "Top borer", "Healthy", "termite"]
        
        for i in range(50):
            pred_class = random.choice(classes)
            confidence = random.uniform(0.5, 0.99)
            image_id = f"test_pred_{i:03d}"
            
            pred_id = tracker.log_prediction(
                image_id=image_id,
                predicted_class=pred_class,
                confidence=confidence,
                device_id=f"device_{i % 3}",
                session_id=f"user_{i % 5}"
            )
            
            # 70% of predictions get feedback
            if random.random() < 0.7:
                is_correct = random.random() < 0.75  # 75% accuracy
                actual = pred_class if is_correct else random.choice([c for c in classes if c != pred_class])
                
                tracker.log_correction(
                    image_id=image_id,
                    actual_class=actual,
                    corrector_id=f"user_{i % 5}",
                    correction_source="test"
                )
        
        # Verify storage
        json_path = Path(temp_dir) / "corrections_log.json"
        db_path = Path(temp_dir) / "corrections.db"
        
        assert json_path.exists(), "JSON file should exist"
        assert db_path.exists(), "SQLite database should exist"
        
        print(f"✅ JSON storage: {json_path} ({json_path.stat().st_size} bytes)")
        print(f"✅ SQLite storage: {db_path} ({db_path.stat().st_size} bytes)")
        
        # Get stats
        stats = tracker.get_stats_summary()
        print(f"✅ Stats summary: {stats}")
        
        print("✅ CorrectionTracker: PASSED")
        return True, temp_dir
        
    except Exception as e:
        print(f"❌ CorrectionTracker: FAILED - {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False, None


def test_performance_analytics(data_dir):
    """Test the PerformanceAnalytics component."""
    print("\n" + "=" * 60)
    print("TEST: PerformanceAnalytics")
    print("=" * 60)
    
    from src.analytics.performance_analytics import PerformanceAnalytics
    
    try:
        analytics = PerformanceAnalytics(data_dir=data_dir)
        
        # Get overall metrics
        overall = analytics.get_overall_metrics()
        print(f"✅ Overall metrics:")
        print(f"   Total: {overall['total_predictions']}")
        print(f"   Accuracy: {overall['accuracy']:.1%}")
        print(f"   Avg Confidence: {overall['avg_confidence']:.1%}")
        
        # Get class metrics
        class_metrics = analytics.get_class_metrics()
        print(f"✅ Class metrics: {len(class_metrics)} classes")
        for cls, metrics in list(class_metrics.items())[:3]:
            print(f"   {cls}: {metrics.accuracy:.1%} accuracy")
        
        # Get confusion matrix
        confusion = analytics.get_confusion_matrix()
        print(f"✅ Confusion matrix: {len(confusion.classes)} classes")
        
        # Get time series
        time_series = analytics.get_time_series(granularity="daily")
        print(f"✅ Time series: {len(time_series)} data points")
        
        # Get improvement trends
        trends = analytics.get_improvement_trends()
        print(f"✅ Trends: {list(trends.keys())}")
        
        print("✅ PerformanceAnalytics: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ PerformanceAnalytics: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_dashboard(data_dir):
    """Test the PerformanceDashboard component."""
    print("\n" + "=" * 60)
    print("TEST: PerformanceDashboard")
    print("=" * 60)
    
    from src.analytics.performance_dashboard import PerformanceDashboard
    
    try:
        dashboard = PerformanceDashboard(data_dir=data_dir)
        
        # Generate charts (if matplotlib available)
        charts = dashboard.generate_all_charts()
        print(f"✅ Charts generated: {len(charts)} files")
        for chart in charts[:3]:
            print(f"   - {Path(chart).name}")
        
        # Generate markdown report
        report = dashboard.generate_markdown_report()
        print(f"✅ Markdown report: {report}")
        
        # Export CSVs
        exports = dashboard.export_all_csv()
        print(f"✅ CSV exports: {len(exports)} files")
        for exp in exports[:3]:
            print(f"   - {Path(exp).name}")
        
        # Full report
        result = dashboard.generate_full_report()
        print(f"✅ Full report generated:")
        print(f"   Charts: {len(result.get('charts', []))}")
        print(f"   Reports: {len(result.get('reports', []))}")
        print(f"   Exports: {len(result.get('exports', []))}")
        
        print("✅ PerformanceDashboard: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ PerformanceDashboard: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test the integration module."""
    print("\n" + "=" * 60)
    print("TEST: Integration Module")
    print("=" * 60)
    
    from src.analytics import integration
    
    temp_dir = tempfile.mkdtemp(prefix="analytics_int_")
    
    try:
        # Initialize
        result = integration.init_analytics(
            model_version="v1.0.6-integration-test",
            data_dir=temp_dir
        )
        assert result, "init_analytics should return True"
        print("✅ init_analytics() succeeded")
        
        # Log prediction
        result = integration.log_prediction(
            prediction_id="int_test_001",
            predicted_class="army worm",
            confidence=0.85,
            predicted_class_id=0
        )
        assert result, "log_prediction should return True"
        print("✅ log_prediction() succeeded")
        
        # Log correction
        result = integration.log_correction(
            prediction_id="int_test_001",
            predicted_class="army worm",
            actual_class="Stalk borer",
            is_correct=False,
            confidence=0.85
        )
        assert result, "log_correction should return True"
        print("✅ log_correction() succeeded")
        
        # Get summary
        summary = integration.get_summary()
        assert "total_predictions" in summary, "Summary should have total_predictions"
        print(f"✅ get_summary(): {summary['total_predictions']} predictions")
        
        print("✅ Integration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Integration: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  ANALYTICS SYSTEM TEST SUITE")
    print("=" * 60)
    
    results = {}
    data_dir = None
    
    # Test 1: CorrectionTracker
    success, data_dir = test_correction_tracker()
    results["CorrectionTracker"] = success
    
    # Test 2: PerformanceAnalytics (uses data from test 1)
    if data_dir:
        results["PerformanceAnalytics"] = test_performance_analytics(data_dir)
    else:
        results["PerformanceAnalytics"] = False
    
    # Test 3: PerformanceDashboard (uses data from test 1)
    if data_dir:
        results["PerformanceDashboard"] = test_performance_dashboard(data_dir)
    else:
        results["PerformanceDashboard"] = False
    
    # Test 4: Integration module
    results["Integration"] = test_integration()
    
    # Cleanup
    if data_dir:
        shutil.rmtree(data_dir, ignore_errors=True)
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
