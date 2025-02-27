#!/usr/bin/env python3
"""
Run the drift detection dashboard.

This script provides a command-line interface to run the drift detection dashboard,
which visualizes concept drift in the migraine prediction data.
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path so we can import the app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.data.drift import DriftDetector, DriftResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_drift_detection(data_path, output_dir=None, window_size=50, threshold=0.01):
    """
    Run drift detection on the data and visualize the results.
    
    Args:
        data_path: Path to the data file
        output_dir: Directory to save the visualization results
        window_size: Size of the sliding window for drift detection
        threshold: Significance level for drift detection
    """
    logger.info(f"Running drift detection on {data_path}")
    logger.info(f"Parameters: window_size={window_size}, threshold={threshold:.3e}")
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape {df.shape}")
        
        # Filter out non-numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df_numeric = df[numeric_cols]
        logger.info(f"Using {len(numeric_cols)} numeric columns: {list(numeric_cols)}")
        
        # Initialize drift detector
        detector = DriftDetector(window_size=window_size, significance_level=threshold)
        
        # Set reference window
        reference_data = df_numeric.iloc[:min(window_size, len(df_numeric))]
        logger.info(f"Initializing reference window with {len(reference_data)} samples")
        detector.initialize_reference(reference_data)
        
        # Process data and detect drift
        drift_points = []
        drift_severities = []
        feature_drifts = {col: 0 for col in numeric_cols}
        timestamps = []
        
        # Create timestamps (for visualization)
        start_date = datetime.now() - timedelta(days=30)
        for i in range(len(df_numeric)):
            timestamps.append((start_date + timedelta(hours=i)).timestamp())
        
        # Process data in sliding windows
        for i in range(window_size, len(df_numeric), max(1, window_size//4)):
            end_idx = min(i + window_size//2, len(df_numeric))
            if end_idx <= i:
                continue
                
            current_window = df_numeric.iloc[i:end_idx]
            
            if len(current_window) < 5:  # Minimum samples to detect drift
                logger.debug(f"Skipping window at index {i} with only {len(current_window)} samples (minimum 5 required)")
                continue
                
            logger.debug(f"Processing window at index {i} with {len(current_window)} samples")
            drift_results = detector.detect_drift(current_window)
            
            # Calculate overall severity as max of feature severities
            severity = 0.0
            drift_detected = False
            
            for result in drift_results:
                if result.detected:
                    drift_detected = True
                    feature_drifts[result.feature] += 1
                    logger.info(f"Drift detected in feature '{result.feature}' at index {i} with severity {result.severity:.3f}, p-value {result.p_value:.3e}, KS statistic {result.statistic:.3f}")
                    if result.severity and result.severity > severity:
                        severity = result.severity
                else:
                    logger.debug(f"No drift detected in feature '{result.feature}' at index {i}, p-value {result.p_value:.3e}, KS statistic {result.statistic:.3f}")
            
            drift_severities.append(severity)
            
            if drift_detected:
                drift_points.append(i)
                logger.info(f"Overall drift detected at index {i} with max severity {severity:.3f}")
        
        # Log summary statistics
        total_drifts = len(drift_points)
        avg_severity = sum(drift_severities) / len(drift_severities) if drift_severities else 0
        max_severity = max(drift_severities) if drift_severities else 0
        
        logger.info(f"Drift detection summary: total_drifts={total_drifts}, avg_severity={avg_severity:.3f}, max_severity={max_severity:.3f}")
        logger.debug(f"Feature drift counts: {feature_drifts}")
        
        # Prepare visualization data
        visualization_data = {
            "timestamps": timestamps[:len(drift_severities)],
            "severities": drift_severities,
            "drift_points": drift_points,
            "feature_drifts": feature_drifts
        }
        
        # Save visualization data if output directory is specified
        if output_dir:
            output_file = os.path.join(output_dir, "drift_visualization_data.json")
            import json
            with open(output_file, "w") as f:
                json.dump(visualization_data, f)
            logger.info(f"Saved visualization data to {output_file}")
            
            # Generate HTML report
            generate_html_report(visualization_data, output_dir)
        
        return visualization_data
        
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}", exc_info=True)
        raise

def generate_html_report(data, output_dir):
    """
    Generate an HTML report for the drift detection results.
    
    Args:
        data: Drift detection visualization data
        output_dir: Directory to save the HTML report
    """
    try:
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Drift Detection Report</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
            <style>
                body {{ padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
                .chart-container {{ background-color: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); padding: 20px; margin-bottom: 20px; }}
                .drift-summary {{ padding: 25px; background-color: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-top: 20px; }}
                .summary-stats {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
                .stat-item {{ text-align: center; }}
                .stat-value {{ display: block; font-size: 28px; font-weight: bold; color: #2196F3; }}
                .stat-label {{ font-size: 14px; color: #666; }}
                .top-features ul {{ list-style: none; padding: 0; margin: 0; background-color: #f9f9f9; border-radius: 6px; }}
                .top-features li {{ padding: 12px 15px; border-bottom: 1px solid #eee; }}
                .top-features li:last-child {{ border-bottom: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="mb-4">Drift Detection Report</h1>
                <p class="text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h3>Drift Severity Over Time</h3>
                            <canvas id="severityChart"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h3>Feature Drift Frequency</h3>
                            <canvas id="featureDriftChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <div class="drift-summary">
                    <h3>Drift Detection Summary</h3>
                    <div class="summary-stats">
                        <div class="stat-item">
                            <span class="stat-value">{len(data["drift_points"])}</span>
                            <span class="stat-label">Total Drifts</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">{max(data["severities"]) if data["severities"] else 0:.3f}</span>
                            <span class="stat-label">Max Severity</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">{sum(data["severities"]) / len(data["severities"]) if data["severities"] else 0:.3f}</span>
                            <span class="stat-label">Avg Severity</span>
                        </div>
                    </div>
                    <div class="top-features">
                        <h4>Most Affected Features:</h4>
                        <ul id="topFeaturesList">
                            {''.join([f'<li>{feature}: {count} drifts</li>' for feature, count in sorted(data["feature_drifts"].items(), key=lambda x: x[1], reverse=True)[:3]])}
                        </ul>
                    </div>
                </div>
            </div>
            
            <script>
                // Format timestamps
                const timestamps = {data["timestamps"]}.map(ts => {{
                    const date = new Date(ts * 1000);
                    return date.toLocaleDateString();
                }});
                
                // Initialize severity chart
                const severityCtx = document.getElementById('severityChart').getContext('2d');
                const severityChart = new Chart(severityCtx, {{
                    type: 'line',
                    data: {{
                        labels: timestamps,
                        datasets: [{{
                            label: 'Drift Severity',
                            data: {data["severities"]},
                            borderColor: '#FF5722',
                            backgroundColor: 'rgba(255, 87, 34, 0.1)',
                            borderWidth: 2,
                            tension: 0.4,
                            fill: true
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Drift Severity Over Time'
                            }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        return `Severity: ${{context.parsed.y.toFixed(3)}}`;
                                    }}
                                }}
                            }},
                            annotation: {{
                                annotations: {{
                                    thresholdLine: {{
                                        type: 'line',
                                        yMin: 0.6,
                                        yMax: 0.6,
                                        borderColor: 'rgba(255, 0, 0, 0.5)',
                                        borderWidth: 1,
                                        borderDash: [5, 5],
                                        label: {{
                                            content: 'Threshold',
                                            enabled: true,
                                            position: 'end'
                                        }}
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 1,
                                title: {{
                                    display: true,
                                    text: 'Severity'
                                }}
                            }},
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Time'
                                }}
                            }}
                        }}
                    }}
                }});

                // Add drift point annotations
                {data["drift_points"]}.forEach((point, index) => {{
                    if (point < timestamps.length) {{
                        severityChart.options.plugins.annotation.annotations[`drift${{index}}`] = {{
                            type: 'line',
                            xMin: point,
                            xMax: point,
                            borderColor: 'rgba(255, 0, 0, 0.7)',
                            borderWidth: 2
                        }};
                    }}
                }});
                severityChart.update();
                
                // Initialize feature drift chart
                const featureCtx = document.getElementById('featureDriftChart').getContext('2d');
                const featureChart = new Chart(featureCtx, {{
                    type: 'bar',
                    data: {{
                        labels: Object.keys({data["feature_drifts"]}),
                        datasets: [{{
                            label: 'Drift Frequency',
                            data: Object.values({data["feature_drifts"]}),
                            backgroundColor: '#2196F3'
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Feature Drift Frequency'
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                title: {{
                                    display: true,
                                    text: 'Frequency'
                                }}
                            }},
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Feature'
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        # Save HTML report
        output_file = os.path.join(output_dir, "drift_report.html")
        with open(output_file, "w") as f:
            f.write(html_content)
        logger.info(f"Generated HTML report at {output_file}")
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Run drift detection dashboard")
    parser.add_argument("--data", required=True, help="Path to the data file")
    parser.add_argument("--output", default="./drift_results", help="Directory to save the results")
    parser.add_argument("--window-size", type=int, default=50, help="Window size for drift detection")
    parser.add_argument("--threshold", type=float, default=0.01, help="Significance level for drift detection")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    run_drift_detection(
        data_path=args.data,
        output_dir=args.output,
        window_size=args.window_size,
        threshold=args.threshold
    )
    
    logger.info(f"Drift detection completed. Results saved to {args.output}")
    logger.info(f"View the HTML report at {os.path.join(args.output, 'drift_report.html')}")

if __name__ == "__main__":
    main()
