"""
PERFORMANCE DASHBOARD
=====================
Generates charts, Markdown reports, CSV exports, and interactive dashboard.

Features:
- Accuracy trend charts (matplotlib)
- Confusion matrix heatmaps
- Per-class bar charts
- Markdown report generation
- CSV data exports
- HTML dashboard with interactive elements

Usage:
    from analytics import PerformanceDashboard

    dashboard = PerformanceDashboard()

    # Generate all reports
    dashboard.generate_full_report(output_dir="reports/2026-01")

    # Individual exports
    dashboard.export_csv("exports/metrics.csv")
    dashboard.generate_charts("charts/")
    dashboard.generate_markdown("reports/report.md")
    dashboard.run_dashboard(port=8050)
"""

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import analytics
from .performance_analytics import (
    ClassMetrics,
    ConfusionData,
    PerformanceAnalytics,
    TimeSeriesMetrics,
)

logger = logging.getLogger(__name__)

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - charts will be disabled")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class PerformanceDashboard:
    """
    Dashboard for generating reports and visualizations.
    """

    def __init__(self, data_dir: str | Path | None = None):
        """
        Initialize dashboard.

        Args:
            data_dir: Directory containing analytics data
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "feedback_data" / "analytics"

        self.data_dir = Path(data_dir)
        self.analytics = PerformanceAnalytics(data_dir)

        # Output directories
        self.reports_dir = self.data_dir / "reports"
        self.charts_dir = self.data_dir / "charts"
        self.exports_dir = self.data_dir / "exports"

        # Ensure directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # CHART GENERATION
    # =========================================================

    def generate_accuracy_trend_chart(
        self,
        output_path: str | Path | None = None,
        granularity: str = "daily",
        last_n_periods: int = 30
    ) -> str | None:
        """
        Generate accuracy trend line chart.

        Returns:
            Path to saved chart, or None if failed
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping chart")
            return None

        time_series = self.analytics.get_time_series(granularity=granularity)

        if not time_series:
            logger.warning("No data for accuracy trend chart")
            return None

        # Take last N periods
        time_series = time_series[-last_n_periods:]

        dates = [ts.date for ts in time_series]
        accuracies = [ts.accuracy * 100 for ts in time_series]
        totals = [ts.total_predictions for ts in time_series]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Accuracy line
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Accuracy (%)', color=color)
        ax1.plot(dates, accuracies, 'b-o', linewidth=2, markersize=4, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 100)

        # Add trend line
        if len(accuracies) >= 3 and NUMPY_AVAILABLE:
            z = np.polyfit(range(len(accuracies)), accuracies, 1)
            p = np.poly1d(z)
            ax1.plot(dates, p(range(len(accuracies))), 'b--', alpha=0.5, label='Trend')

        # Prediction count bars
        ax2 = ax1.twinx()
        color = 'tab:gray'
        ax2.set_ylabel('Predictions', color=color)
        ax2.bar(dates, totals, alpha=0.3, color=color, label='Predictions')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'Accuracy Trend ({granularity.capitalize()})')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = self.charts_dir / f"accuracy_trend_{granularity}.png"

        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved accuracy trend chart: {output_path}")
        return str(output_path)

    def generate_confusion_matrix_chart(
        self,
        output_path: str | Path | None = None,
        start_date: str | None = None,
        end_date: str | None = None
    ) -> str | None:
        """
        Generate confusion matrix heatmap.

        Returns:
            Path to saved chart, or None if failed
        """
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("matplotlib/numpy not available, skipping chart")
            return None

        confusion = self.analytics.get_confusion_matrix(
            start_date=start_date,
            end_date=end_date
        )

        if not confusion.classes:
            logger.warning("No data for confusion matrix")
            return None

        matrix = np.array(confusion.matrix)
        classes = confusion.classes

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(matrix, cmap='Blues')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')

        # Labels
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)

        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Actual Class')
        ax.set_title('Confusion Matrix')

        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                value = matrix[i, j]
                if value > 0:
                    color = 'white' if value > matrix.max() / 2 else 'black'
                    ax.text(j, i, str(value), ha='center', va='center', color=color, fontsize=8)

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = self.charts_dir / "confusion_matrix.png"

        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved confusion matrix chart: {output_path}")
        return str(output_path)

    def generate_class_performance_chart(
        self,
        output_path: str | Path | None = None,
        metric: str = "accuracy"  # accuracy, precision, recall, f1
    ) -> str | None:
        """
        Generate per-class performance bar chart.

        Returns:
            Path to saved chart, or None if failed
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping chart")
            return None

        class_metrics = self.analytics.get_class_metrics()

        if not class_metrics:
            logger.warning("No data for class performance chart")
            return None

        # Sort by metric value
        classes = list(class_metrics.keys())
        if metric == "accuracy":
            values = [class_metrics[c].accuracy * 100 for c in classes]
        elif metric == "precision":
            values = [class_metrics[c].precision * 100 for c in classes]
        elif metric == "recall":
            values = [class_metrics[c].recall * 100 for c in classes]
        else:  # f1
            values = [class_metrics[c].f1_score * 100 for c in classes]

        # Sort by value
        sorted_pairs = sorted(zip(classes, values), key=lambda x: -x[1])
        classes = [p[0] for p in sorted_pairs]
        values = [p[1] for p in sorted_pairs]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['green' if v >= 80 else 'orange' if v >= 60 else 'red' for v in values]
        bars = ax.barh(classes, values, color=colors, alpha=0.8)

        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}%', va='center', fontsize=9)

        ax.set_xlim(0, 105)
        ax.set_xlabel(f'{metric.capitalize()} (%)')
        ax.set_title(f'Per-Class {metric.capitalize()}')
        ax.invert_yaxis()  # Top to bottom

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = self.charts_dir / f"class_{metric}.png"

        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved class performance chart: {output_path}")
        return str(output_path)

    def generate_correction_frequency_chart(
        self,
        output_path: str | Path | None = None,
        top_n: int = 10
    ) -> str | None:
        """
        Generate chart of most common misclassifications.

        Returns:
            Path to saved chart, or None if failed
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping chart")
            return None

        corrections = self.analytics.get_correction_frequency(top_n=top_n)

        if not corrections:
            logger.warning("No correction data for chart")
            return None

        labels = [f"{c['predicted']} → {c['actual']}" for c in corrections]
        counts = [c['count'] for c in corrections]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(labels, counts, color='coral', alpha=0.8)

        for i, (label, count) in enumerate(zip(labels, counts)):
            ax.text(count + 0.5, i, str(count), va='center')

        ax.set_xlabel('Correction Count')
        ax.set_title('Most Common Misclassifications')
        ax.invert_yaxis()

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = self.charts_dir / "correction_frequency.png"

        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved correction frequency chart: {output_path}")
        return str(output_path)

    def generate_model_comparison_chart(
        self,
        output_path: str | Path | None = None
    ) -> str | None:
        """
        Generate model version comparison chart.

        Returns:
            Path to saved chart, or None if failed
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping chart")
            return None

        comparison = self.analytics.get_model_comparison()

        if not comparison:
            logger.warning("No model comparison data")
            return None

        versions = list(comparison.keys())
        accuracies = [comparison[v]['accuracy'] * 100 for v in versions]
        totals = [comparison[v]['total_predictions'] for v in versions]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        x = range(len(versions))
        width = 0.35

        # Accuracy bars
        bars1 = ax1.bar([i - width/2 for i in x], accuracies, width,
                       label='Accuracy', color='steelblue', alpha=0.8)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)

        # Prediction count on secondary axis
        ax2 = ax1.twinx()
        bars2 = ax2.bar([i + width/2 for i in x], totals, width,
                       label='Predictions', color='gray', alpha=0.5)
        ax2.set_ylabel('Predictions')

        ax1.set_xticks(x)
        ax1.set_xticklabels(versions, rotation=45, ha='right')
        ax1.set_xlabel('Model Version')
        ax1.set_title('Model Version Comparison')

        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()

        # Save
        if output_path is None:
            output_path = self.charts_dir / "model_comparison.png"

        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved model comparison chart: {output_path}")
        return str(output_path)

    def generate_all_charts(self, output_dir: str | Path | None = None) -> list[str]:
        """
        Generate all available charts.

        Returns:
            List of paths to generated charts
        """
        if output_dir:
            self.charts_dir = Path(output_dir)
            self.charts_dir.mkdir(parents=True, exist_ok=True)

        charts = []

        # Generate each chart type
        chart = self.generate_accuracy_trend_chart(granularity="daily")
        if chart:
            charts.append(chart)

        chart = self.generate_accuracy_trend_chart(granularity="weekly")
        if chart:
            charts.append(chart)

        chart = self.generate_confusion_matrix_chart()
        if chart:
            charts.append(chart)

        chart = self.generate_class_performance_chart(metric="accuracy")
        if chart:
            charts.append(chart)

        chart = self.generate_class_performance_chart(metric="precision")
        if chart:
            charts.append(chart)

        chart = self.generate_class_performance_chart(metric="recall")
        if chart:
            charts.append(chart)

        chart = self.generate_correction_frequency_chart()
        if chart:
            charts.append(chart)

        chart = self.generate_model_comparison_chart()
        if chart:
            charts.append(chart)

        logger.info(f"Generated {len(charts)} charts")
        return charts

    # =========================================================
    # MARKDOWN REPORT GENERATION
    # =========================================================

    def generate_markdown_report(
        self,
        output_path: str | Path | None = None,
        include_charts: bool = True
    ) -> str:
        """
        Generate comprehensive Markdown report.

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Gather data
        overall = self.analytics.get_overall_metrics()
        class_metrics = self.analytics.get_class_metrics()
        trends = self.analytics.get_improvement_trends()
        corrections = self.analytics.get_correction_frequency(top_n=10)
        model_comparison = self.analytics.get_model_comparison()

        # Build report
        lines = [
            "# Intelli-PEST Performance Report",
            "",
            f"**Generated:** {timestamp}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Predictions | {overall['total_predictions']:,} |",
            f"| Correct | {overall['correct']:,} |",
            f"| Incorrect | {overall['incorrect']:,} |",
            f"| **Overall Accuracy** | **{overall['accuracy']:.1%}** |",
            f"| Avg Confidence | {overall['avg_confidence']:.1%} |",
            f"| Avg Confidence (Correct) | {overall['avg_confidence_correct']:.1%} |",
            f"| Avg Confidence (Incorrect) | {overall['avg_confidence_incorrect']:.1%} |",
            "",
        ]

        # Trend summary
        if 'overall_trend' in trends:
            trend = trends['overall_trend']
            emoji = "📈" if trend['trend'] == 'improving' else "📉" if trend['trend'] == 'declining' else "➡️"
            lines.extend([
                f"### Trend Analysis {emoji}",
                "",
                f"- **Recent Accuracy:** {trend['recent_accuracy']:.1%}",
                f"- **Earlier Accuracy:** {trend['earlier_accuracy']:.1%}",
                f"- **Change:** {trend['improvement']:+.1%} ({trend['trend']})",
                "",
            ])

        # Include chart references
        if include_charts:
            lines.extend([
                "## Accuracy Trend",
                "",
                "![Accuracy Trend](charts/accuracy_trend_daily.png)",
                "",
            ])

        # Per-class performance
        lines.extend([
            "## Per-Class Performance",
            "",
            "| Class | Predictions | Accuracy | Precision | Recall | F1 |",
            "|-------|-------------|----------|-----------|--------|-----|",
        ])

        for cls, metrics in sorted(class_metrics.items(), key=lambda x: -x[1].accuracy):
            lines.append(
                f"| {cls} | {metrics.total_predictions} | "
                f"{metrics.accuracy:.1%} | {metrics.precision:.1%} | "
                f"{metrics.recall:.1%} | {metrics.f1_score:.2f} |"
            )

        lines.append("")

        if include_charts:
            lines.extend([
                "![Class Accuracy](charts/class_accuracy.png)",
                "",
            ])

        # Confusion matrix
        if include_charts:
            lines.extend([
                "## Confusion Matrix",
                "",
                "![Confusion Matrix](charts/confusion_matrix.png)",
                "",
            ])

        # Most common corrections
        if corrections:
            lines.extend([
                "## Most Common Misclassifications",
                "",
                "| Predicted | Actual | Count |",
                "|-----------|--------|-------|",
            ])

            for c in corrections[:10]:
                lines.append(f"| {c['predicted']} | {c['actual']} | {c['count']} |")

            lines.append("")

            if include_charts:
                lines.extend([
                    "![Correction Frequency](charts/correction_frequency.png)",
                    "",
                ])

        # Class improvement trends
        if 'top_improving' in trends and trends['top_improving']:
            lines.extend([
                "## Class Improvement Trends",
                "",
                "### Top Improving Classes 📈",
                "",
            ])

            for cls, improvement in trends['top_improving'][:5]:
                emoji = "🟢" if improvement > 0.05 else "🟡" if improvement > 0 else "🔴"
                lines.append(f"- {emoji} **{cls}**: {improvement:+.1%}")

            lines.append("")

            lines.append("### Classes Needing Attention 📉")
            lines.append("")

            for cls, improvement in trends['needs_attention'][:5]:
                emoji = "🔴" if improvement < -0.05 else "🟡" if improvement < 0 else "🟢"
                lines.append(f"- {emoji} **{cls}**: {improvement:+.1%}")

            lines.append("")

        # Model version comparison
        if model_comparison:
            lines.extend([
                "## Model Version Comparison",
                "",
                "| Version | Predictions | Accuracy | Avg Confidence | Active Days |",
                "|---------|-------------|----------|----------------|-------------|",
            ])

            for version, data in model_comparison.items():
                lines.append(
                    f"| {version} | {data['total_predictions']:,} | "
                    f"{data['accuracy']:.1%} | {data['avg_confidence']:.1%} | "
                    f"{data['active_days']} |"
                )

            lines.append("")

            if include_charts:
                lines.extend([
                    "![Model Comparison](charts/model_comparison.png)",
                    "",
                ])

        # Confidence calibration
        if 'confidence_calibration' in overall:
            lines.extend([
                "## Confidence Calibration",
                "",
                "| Confidence Bin | Avg Confidence | Actual Accuracy | Gap | Count |",
                "|----------------|----------------|-----------------|-----|-------|",
            ])

            for bin_name, data in overall['confidence_calibration'].items():
                gap_emoji = "✓" if data['gap'] < 0.1 else "⚠️"
                lines.append(
                    f"| {bin_name} | {data['avg_confidence']:.1%} | "
                    f"{data['actual_accuracy']:.1%} | {data['gap']:.1%} {gap_emoji} | "
                    f"{data['count']} |"
                )

            lines.append("")

        # Footer
        lines.extend([
            "---",
            "",
            "*Report generated by Intelli-PEST Analytics System*",
        ])

        # Write report
        if output_path is None:
            output_path = self.reports_dir / f"performance_report_{date_str}.md"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Generated Markdown report: {output_path}")
        return str(output_path)

    # =========================================================
    # CSV EXPORTS
    # =========================================================

    def export_overall_metrics_csv(self, output_path: str | Path | None = None) -> str:
        """Export overall metrics to CSV."""
        overall = self.analytics.get_overall_metrics()

        if output_path is None:
            output_path = self.exports_dir / "overall_metrics.csv"

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in overall.items():
                if not isinstance(value, dict):
                    writer.writerow([key, value])

        logger.info(f"Exported overall metrics: {output_path}")
        return str(output_path)

    def export_class_metrics_csv(self, output_path: str | Path | None = None) -> str:
        """Export per-class metrics to CSV."""
        class_metrics = self.analytics.get_class_metrics()

        if output_path is None:
            output_path = self.exports_dir / "class_metrics.csv"

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Class', 'Total Predictions', 'Correct', 'Incorrect',
                'Accuracy', 'Precision', 'Recall', 'F1 Score',
                'Avg Confidence', 'Avg Conf (Correct)', 'Avg Conf (Incorrect)'
            ])

            for cls, metrics in sorted(class_metrics.items()):
                writer.writerow([
                    cls, metrics.total_predictions, metrics.correct, metrics.incorrect,
                    f"{metrics.accuracy:.4f}", f"{metrics.precision:.4f}",
                    f"{metrics.recall:.4f}", f"{metrics.f1_score:.4f}",
                    f"{metrics.avg_confidence:.4f}",
                    f"{metrics.avg_confidence_when_correct:.4f}",
                    f"{metrics.avg_confidence_when_wrong:.4f}"
                ])

        logger.info(f"Exported class metrics: {output_path}")
        return str(output_path)

    def export_time_series_csv(
        self,
        output_path: str | Path | None = None,
        granularity: str = "daily"
    ) -> str:
        """Export time series metrics to CSV."""
        time_series = self.analytics.get_time_series(granularity=granularity)

        if output_path is None:
            output_path = self.exports_dir / f"time_series_{granularity}.csv"

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Date', 'Total Predictions', 'Correct', 'Incorrect',
                'Pending', 'Accuracy', 'Avg Confidence', 'Model Version'
            ])

            for ts in time_series:
                writer.writerow([
                    ts.date, ts.total_predictions, ts.correct, ts.incorrect,
                    ts.pending, f"{ts.accuracy:.4f}", f"{ts.avg_confidence:.4f}",
                    ts.model_version
                ])

        logger.info(f"Exported time series: {output_path}")
        return str(output_path)

    def export_confusion_matrix_csv(self, output_path: str | Path | None = None) -> str:
        """Export confusion matrix to CSV."""
        confusion = self.analytics.get_confusion_matrix()

        if output_path is None:
            output_path = self.exports_dir / "confusion_matrix.csv"

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header row with class names
            writer.writerow(['Actual \\ Predicted'] + confusion.classes)

            # Matrix rows
            for i, actual_class in enumerate(confusion.classes):
                writer.writerow([actual_class] + confusion.matrix[i])

        logger.info(f"Exported confusion matrix: {output_path}")
        return str(output_path)

    def export_corrections_csv(self, output_path: str | Path | None = None) -> str:
        """Export correction frequency to CSV."""
        corrections = self.analytics.get_correction_frequency(top_n=100)

        if output_path is None:
            output_path = self.exports_dir / "corrections.csv"

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Predicted Class', 'Actual Class', 'Count', 'Description'])

            for c in corrections:
                writer.writerow([c['predicted'], c['actual'], c['count'], c['description']])

        logger.info(f"Exported corrections: {output_path}")
        return str(output_path)

    def export_all_csv(self, output_dir: str | Path | None = None) -> list[str]:
        """Export all data to CSV files."""
        if output_dir:
            self.exports_dir = Path(output_dir)
            self.exports_dir.mkdir(parents=True, exist_ok=True)

        exports = []
        exports.append(self.export_overall_metrics_csv())
        exports.append(self.export_class_metrics_csv())
        exports.append(self.export_time_series_csv(granularity="daily"))
        exports.append(self.export_time_series_csv(granularity="weekly"))
        exports.append(self.export_confusion_matrix_csv())
        exports.append(self.export_corrections_csv())

        logger.info(f"Exported {len(exports)} CSV files")
        return exports

    # =========================================================
    # FULL REPORT GENERATION
    # =========================================================

    def generate_full_report(
        self,
        output_dir: str | Path | None = None,
        include_charts: bool = True,
        include_csv: bool = True
    ) -> dict[str, Any]:
        """
        Generate complete report with charts, Markdown, and CSV exports.

        Returns:
            Dictionary with paths to all generated files
        """
        if output_dir:
            base_dir = Path(output_dir)
            self.reports_dir = base_dir / "reports"
            self.charts_dir = base_dir / "charts"
            self.exports_dir = base_dir / "exports"

            self.reports_dir.mkdir(parents=True, exist_ok=True)
            self.charts_dir.mkdir(parents=True, exist_ok=True)
            self.exports_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "generated_at": datetime.now().isoformat(),
            "charts": [],
            "reports": [],
            "exports": []
        }

        # Generate charts
        if include_charts:
            result["charts"] = self.generate_all_charts()

        # Generate Markdown report
        result["reports"].append(self.generate_markdown_report(include_charts=include_charts))

        # Generate CSV exports
        if include_csv:
            result["exports"] = self.export_all_csv()

        # Save manifest
        manifest_path = (Path(output_dir) if output_dir else self.data_dir) / "report_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Generated full report: {len(result['charts'])} charts, "
                   f"{len(result['reports'])} reports, {len(result['exports'])} exports")

        return result


# =========================================================
# CONVENIENCE FUNCTIONS
# =========================================================

def generate_report(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Convenience function to generate full report."""
    dashboard = PerformanceDashboard()
    return dashboard.generate_full_report(output_dir=output_dir)


def generate_charts(output_dir: str | Path | None = None) -> list[str]:
    """Convenience function to generate all charts."""
    dashboard = PerformanceDashboard()
    return dashboard.generate_all_charts(output_dir=output_dir)


def export_csv(output_dir: str | Path | None = None) -> list[str]:
    """Convenience function to export all CSV files."""
    dashboard = PerformanceDashboard()
    return dashboard.export_all_csv(output_dir=output_dir)
