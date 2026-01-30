#!/usr/bin/env python3
"""
ANALYTICS CLI
=============
Command-line interface for generating performance reports and analytics.

Usage:
    python -m src.analytics.cli report         # Full report
    python -m src.analytics.cli charts         # Charts only
    python -m src.analytics.cli csv            # CSV exports only
    python -m src.analytics.cli summary        # Print summary to console
    python -m src.analytics.cli dashboard      # Launch interactive dashboard
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def print_summary():
    """Print quick summary to console."""
    from .performance_analytics import PerformanceAnalytics
    
    analytics = PerformanceAnalytics()
    overall = analytics.get_overall_metrics()
    class_metrics = analytics.get_class_metrics()
    trends = analytics.get_improvement_trends()
    
    print("\n" + "=" * 60)
    print("  INTELLI-PEST PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 OVERALL METRICS")
    print(f"   Total Predictions: {overall['total_predictions']:,}")
    print(f"   Correct: {overall['correct']:,}")
    print(f"   Incorrect: {overall['incorrect']:,}")
    print(f"   ✅ Accuracy: {overall['accuracy']:.1%}")
    print(f"   🎯 Avg Confidence: {overall['avg_confidence']:.1%}")
    
    if 'overall_trend' in trends:
        trend = trends['overall_trend']
        if trend['trend'] == 'improving':
            emoji = "📈"
        elif trend['trend'] == 'declining':
            emoji = "📉"
        else:
            emoji = "➡️"
        print(f"\n{emoji} TREND: {trend['trend'].upper()}")
        print(f"   Recent: {trend['recent_accuracy']:.1%}")
        print(f"   Earlier: {trend['earlier_accuracy']:.1%}")
        print(f"   Change: {trend['improvement']:+.1%}")
    
    print(f"\n📋 PER-CLASS ACCURACY (Top 5)")
    sorted_classes = sorted(class_metrics.items(), key=lambda x: -x[1].accuracy)
    for cls, metrics in sorted_classes[:5]:
        status = "🟢" if metrics.accuracy >= 0.8 else "🟡" if metrics.accuracy >= 0.6 else "🔴"
        print(f"   {status} {cls}: {metrics.accuracy:.1%} ({metrics.total_predictions} predictions)")
    
    if len(sorted_classes) > 5:
        print(f"\n📋 NEEDS ATTENTION (Bottom 3)")
        for cls, metrics in sorted_classes[-3:]:
            status = "🔴" if metrics.accuracy < 0.6 else "🟡" if metrics.accuracy < 0.8 else "🟢"
            print(f"   {status} {cls}: {metrics.accuracy:.1%} ({metrics.total_predictions} predictions)")
    
    print("\n" + "=" * 60)
    print()


def generate_full_report(output_dir: Optional[str] = None):
    """Generate comprehensive report."""
    from .performance_dashboard import PerformanceDashboard
    
    dashboard = PerformanceDashboard()
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(__file__).parent.parent.parent / "feedback_data" / "analytics" / f"report_{timestamp}")
    
    print(f"\n📊 Generating full report...")
    print(f"   Output: {output_dir}")
    
    result = dashboard.generate_full_report(output_dir=str(output_dir))
    
    print(f"\n✅ Report generated successfully!")
    print(f"   📈 Charts: {len(result['charts'])} files")
    print(f"   📄 Reports: {len(result['reports'])} files")
    print(f"   📋 Exports: {len(result['exports'])} files")
    
    # Print file locations
    if result['reports']:
        print(f"\n📄 Main report: {result['reports'][0]}")
    
    return result


def generate_charts_only(output_dir: Optional[str] = None):
    """Generate charts only."""
    from .performance_dashboard import PerformanceDashboard
    
    dashboard = PerformanceDashboard()
    
    if output_dir:
        dashboard.charts_dir = Path(output_dir)
        dashboard.charts_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generating charts...")
    charts = dashboard.generate_all_charts()
    
    print(f"\n✅ Generated {len(charts)} charts:")
    for chart in charts:
        print(f"   📈 {chart}")
    
    return charts


def export_csv_only(output_dir: Optional[str] = None):
    """Export CSV files only."""
    from .performance_dashboard import PerformanceDashboard
    
    dashboard = PerformanceDashboard()
    
    if output_dir:
        dashboard.exports_dir = Path(output_dir)
        dashboard.exports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📋 Exporting CSV files...")
    exports = dashboard.export_all_csv()
    
    print(f"\n✅ Exported {len(exports)} CSV files:")
    for export in exports:
        print(f"   📄 {export}")
    
    return exports


def launch_dashboard(port: int = 8050):
    """Launch interactive HTML dashboard."""
    from .performance_analytics import PerformanceAnalytics
    from .performance_dashboard import PerformanceDashboard
    import webbrowser
    import http.server
    import threading
    
    # Generate latest charts
    dashboard = PerformanceDashboard()
    dashboard.generate_all_charts()
    dashboard.generate_markdown_report()
    
    # Create simple HTML dashboard
    html_content = create_dashboard_html(dashboard)
    
    html_path = dashboard.data_dir / "dashboard.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n🌐 Dashboard generated: {html_path}")
    print(f"   Opening in browser...")
    
    webbrowser.open(f"file://{html_path}")


def create_dashboard_html(dashboard) -> str:
    """Create interactive HTML dashboard."""
    analytics = dashboard.analytics
    overall = analytics.get_overall_metrics()
    class_metrics = analytics.get_class_metrics()
    trends = analytics.get_improvement_trends()
    
    # Build class rows
    class_rows = ""
    for cls, metrics in sorted(class_metrics.items(), key=lambda x: -x[1].accuracy):
        status_class = "good" if metrics.accuracy >= 0.8 else "warning" if metrics.accuracy >= 0.6 else "bad"
        class_rows += f"""
        <tr class="{status_class}">
            <td>{cls}</td>
            <td>{metrics.total_predictions}</td>
            <td>{metrics.accuracy:.1%}</td>
            <td>{metrics.precision:.1%}</td>
            <td>{metrics.recall:.1%}</td>
            <td>{metrics.f1_score:.2f}</td>
        </tr>
        """
    
    # Trend info
    trend_html = ""
    if 'overall_trend' in trends:
        trend = trends['overall_trend']
        trend_class = "improving" if trend['trend'] == 'improving' else "declining" if trend['trend'] == 'declining' else "stable"
        trend_html = f"""
        <div class="trend-card {trend_class}">
            <h3>Trend: {trend['trend'].upper()}</h3>
            <p>Recent: {trend['recent_accuracy']:.1%} | Earlier: {trend['earlier_accuracy']:.1%}</p>
            <p>Change: {trend['improvement']:+.1%}</p>
        </div>
        """
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelli-PEST Performance Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, #2d5016 0%, #4a7c23 100%);
            color: white;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        header h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        header p {{ opacity: 0.9; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-card h3 {{ color: #666; font-size: 0.9rem; margin-bottom: 10px; }}
        .metric-card .value {{ font-size: 2rem; font-weight: bold; color: #2d5016; }}
        .metric-card.accuracy .value {{ color: #4a7c23; }}
        
        .trend-card {{
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .trend-card.improving {{ background: #d4edda; color: #155724; }}
        .trend-card.declining {{ background: #f8d7da; color: #721c24; }}
        .trend-card.stable {{ background: #fff3cd; color: #856404; }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-card h3 {{ margin-bottom: 15px; color: #333; }}
        .chart-card img {{ width: 100%; height: auto; border-radius: 5px; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th, td {{ padding: 12px 15px; text-align: left; }}
        th {{ background: #2d5016; color: white; }}
        tr:nth-child(even) {{ background: #f8f8f8; }}
        tr.good td:nth-child(3) {{ color: #155724; font-weight: bold; }}
        tr.warning td:nth-child(3) {{ color: #856404; font-weight: bold; }}
        tr.bad td:nth-child(3) {{ color: #721c24; font-weight: bold; }}
        
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ 
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #4a7c23;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌿 Intelli-PEST Dashboard</h1>
            <p>Performance Correction Tracking System</p>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </header>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Predictions</h3>
                <div class="value">{overall['total_predictions']:,}</div>
            </div>
            <div class="metric-card accuracy">
                <h3>Overall Accuracy</h3>
                <div class="value">{overall['accuracy']:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Correct</h3>
                <div class="value">{overall['correct']:,}</div>
            </div>
            <div class="metric-card">
                <h3>Incorrect</h3>
                <div class="value">{overall['incorrect']:,}</div>
            </div>
            <div class="metric-card">
                <h3>Avg Confidence</h3>
                <div class="value">{overall['avg_confidence']:.1%}</div>
            </div>
        </div>
        
        {trend_html}
        
        <div class="section">
            <h2>📈 Performance Charts</h2>
            <div class="charts-grid">
                <div class="chart-card">
                    <h3>Accuracy Trend</h3>
                    <img src="charts/accuracy_trend_daily.png" alt="Accuracy Trend" onerror="this.style.display='none'">
                </div>
                <div class="chart-card">
                    <h3>Confusion Matrix</h3>
                    <img src="charts/confusion_matrix.png" alt="Confusion Matrix" onerror="this.style.display='none'">
                </div>
                <div class="chart-card">
                    <h3>Class Accuracy</h3>
                    <img src="charts/class_accuracy.png" alt="Class Accuracy" onerror="this.style.display='none'">
                </div>
                <div class="chart-card">
                    <h3>Common Misclassifications</h3>
                    <img src="charts/correction_frequency.png" alt="Corrections" onerror="this.style.display='none'">
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Per-Class Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Predictions</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                    </tr>
                </thead>
                <tbody>
                    {class_rows}
                </tbody>
            </table>
        </div>
        
        <footer>
            <p>Intelli-PEST Analytics System | Performance Correction Tracking</p>
        </footer>
    </div>
</body>
</html>"""
    
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Intelli-PEST Analytics CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.analytics.cli summary     Print quick summary
  python -m src.analytics.cli report      Generate full report
  python -m src.analytics.cli charts      Generate charts only
  python -m src.analytics.cli csv         Export CSV files
  python -m src.analytics.cli dashboard   Open interactive dashboard
        """
    )
    
    parser.add_argument(
        "command",
        choices=["summary", "report", "charts", "csv", "dashboard"],
        help="Command to run"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output directory for generated files"
    )
    
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8050,
        help="Port for dashboard server (default: 8050)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "summary":
            print_summary()
        
        elif args.command == "report":
            generate_full_report(args.output)
        
        elif args.command == "charts":
            generate_charts_only(args.output)
        
        elif args.command == "csv":
            export_csv_only(args.output)
        
        elif args.command == "dashboard":
            launch_dashboard(args.port)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
