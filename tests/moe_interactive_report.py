import os
import sys
import json
import pandas as pd
import numpy as np
import logging
import math
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_interactive_report(test_results, results_dir):
    """Generate an interactive HTML report with Plotly visualizations.
    
    Args:
        test_results: Dictionary of test results
        results_dir: Directory with result files
        
    Returns:
        Path to the generated HTML report
    """
    logger.info("Generating interactive HTML report...")
    
    # Create report directory if it doesn't exist
    results_dir = str(results_dir) if not isinstance(results_dir, str) else results_dir
    
    # Convert to absolute path if it's a relative path
    if not os.path.isabs(results_dir):
        # Get the absolute path of the current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Resolve the relative path from the current directory
        results_dir = os.path.abspath(os.path.join(current_dir, results_dir))
    
    reports_dir = os.path.join(results_dir, "reports")
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    
    # Define report path
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(reports_dir, f"interactive_report_{timestamp}.html")
    
    logger.info(f"Will save interactive report to absolute path: {os.path.abspath(report_path)}")
    
    # HTML container
    html_content = []
    html_content.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MoE Validation Interactive Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1, h2, h3, h4 {
                color: #2c3e50;
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eee;
            }
            .plot-container {
                margin: 30px 0;
                background-color: white;
                padding: 15px;
                border-radius: 4px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.05);
            }
            .notification {
                padding: 15px;
                background-color: #e3f2fd;
                border-left: 4px solid #2196F3;
                margin-bottom: 20px;
            }
            .notification.warning {
                background-color: #fff8e1;
                border-left-color: #ffc107;
            }
            .notification.critical {
                background-color: #ffebee;
                border-left-color: #f44336;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
            }
            th {
                background-color: #f4f7f9;
                font-weight: 500;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            tr:hover {
                background-color: #f1f1f1;
            }
            .summary-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }
            .summary-card {
                flex: 1;
                background-color: white;
                border-radius: 4px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                min-width: 200px;
            }
            .summary-card h3 {
                margin-top: 0;
                font-size: 16px;
                color: #555;
            }
            .summary-value {
                font-size: 24px;
                font-weight: 500;
                margin: 10px 0;
            }
            .indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .green {
                background-color: #4CAF50;
            }
            .red {
                background-color: #F44336;
            }
            .amber {
                background-color: #FFC107;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MoE Validation Interactive Report</h1>
            <p>Report generated on: """ + timestamp.replace("_", " ") + """</p>
    """)
    
    # Add summary section
    test_count = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result.get('passed', False))
    
    html_content.append(f"""
            <div class="summary-container">
                <div class="summary-card">
                    <h3>Tests Run</h3>
                    <div class="summary-value">{test_count}</div>
                </div>
                <div class="summary-card">
                    <h3>Tests Passed</h3>
                    <div class="summary-value">{passed_tests}</div>
                </div>
                <div class="summary-card">
                    <h3>Success Rate</h3>
                    <div class="summary-value">{(passed_tests/test_count)*100:.1f}%</div>
                </div>
            </div>
    """)
    
    # Add notification section
    if 'drift_detected' in test_results and test_results['drift_detected'].get('passed', False):
        html_content.append("""
            <div class="notification warning">
                <h3>⚠️ Drift Detected</h3>
                <p>Significant data drift has been detected. Review the drift analysis section for details on affected features and potential impacts.</p>
            </div>
        """)
    
    # Add test result table
    html_content.append("""
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
    """)
    
    for test_name, result in test_results.items():
        status = "Passed" if result.get('passed', False) else "Failed"
        indicator_class = "green" if result.get('passed', False) else "red"
        details = result.get('details', '')
        
        html_content.append(f"""
                <tr>
                    <td>{test_name}</td>
                    <td><span class="indicator {indicator_class}"></span>{status}</td>
                    <td>{details}</td>
                </tr>
        """)
    
    html_content.append("""
            </table>
    """)
    
    # Add interactive visualizations
    html_content.append("""
            <h2>Drift Analysis Visualizations</h2>
    """)
    
    # 1. Feature Importance Drift Visualization
    f_importance_path = os.path.join(results_dir, "drift_feature_importance.csv")
    if os.path.exists(f_importance_path):
        try:
            importance_df = pd.read_csv(f_importance_path)
            
            # Create interactive plot
            html_content.append("""
            <div class="plot-container">
                <h3>Feature Importance Drift Analysis</h3>
                <div id="importancePlot" style="height: 500px;"></div>
                <script>
            """)
            
            # Check which format the CSV file has
            if 'before_importance' in importance_df.columns and 'after_importance' in importance_df.columns:
                # Original expected format with before/after columns
                features = importance_df['feature'].tolist()
                before_imp = importance_df['before_importance'].tolist()
                after_imp = importance_df['after_importance'].tolist()
                
                if 'absolute_diff' in importance_df.columns:
                    abs_diff = importance_df['absolute_diff'].tolist()
                else:
                    # Calculate if not present
                    abs_diff = [abs(a - b) for a, b in zip(before_imp, after_imp)]
            else:
                # Alternative format with just importance_shift
                features = importance_df['feature'].tolist()
                
                # Use importance_shift as the main metric
                if 'importance_shift' in importance_df.columns:
                    shift_values = importance_df['importance_shift'].tolist()
                    
                    # For visualization purposes, create dummy before/after values
                    # that illustrate the shift
                    before_imp = [0.5 for _ in features]  # Baseline
                    after_imp = [0.5 + shift for shift in shift_values]  # Apply shift
                    abs_diff = [abs(s) for s in shift_values]  # Absolute shift
                else:
                    # Fallback if we don't recognize the format
                    logger.warning(f"Unrecognized feature importance CSV format")
                    raise ValueError("CSV format not recognized")
            
            # Plot script
            # Check which column format is used
            has_shift_column = 'importance_shift' in importance_df.columns
            
            html_content.append(f"""
                    var features = {features};
                    var beforeImp = {before_imp};
                    var afterImp = {after_imp};
                    var absDiff = {abs_diff};
                    var hasShiftColumn = {str(has_shift_column).lower()};
                    
                    var trace1, trace2, trace3, data, layout;
                    
                    if (hasShiftColumn) {{
                        // For importance shift format
                        trace1 = {{
                            x: features,
                            y: absDiff,
                            name: 'Feature Importance Shift',
                            type: 'bar',
                            marker: {{
                                color: absDiff,
                                colorscale: 'RdBu',
                                showscale: true,
                                colorbar: {{
                                    title: 'Importance Shift Magnitude'
                                }}
                            }}
                        }};
                        
                        data = [trace1];
                        
                        layout = {{
                            title: 'Feature Importance Shift During Drift',
                            xaxis: {{title: 'Features'}},
                            yaxis: {{title: 'Importance Shift Magnitude'}}
                        }};
                    }} else {{
                        // For before/after format
                        trace1 = {{
                            x: features,
                            y: beforeImp,
                            name: 'Before Drift',
                            type: 'bar',
                            marker: {{color: 'royalblue'}}
                        }};
                        
                        trace2 = {{
                            x: features,
                            y: afterImp,
                            name: 'After Drift',
                            type: 'bar',
                            marker: {{color: 'firebrick'}}
                        }};
                        
                        trace3 = {{
                            x: features,
                            y: absDiff,
                            name: 'Absolute Difference',
                            type: 'bar',
                            marker: {{
                                color: absDiff,
                                colorscale: 'YlOrRd',
                                showscale: true
                            }},
                            visible: false
                        }};
                        
                        data = [trace1, trace2, trace3];
                        
                        layout = {{
                            barmode: 'group',
                            title: 'Feature Importance Comparison',
                            xaxis: {{title: 'Features'}},
                            yaxis: {{title: 'Importance Value'}},
                            updatemenus: [{{
                                type: 'buttons',
                                direction: 'left',
                                buttons: [
                                    {{
                                        args: [{{visible: [true, true, false]}}],
                                        label: 'Before vs After',
                                        method: 'update'
                                    }},
                                    {{
                                        args: [{{visible: [false, false, true]}}],
                                        label: 'Absolute Difference',
                                        method: 'update'
                                    }}
                                ],
                                pad: {{'r': 10, 't': 10}},
                                showactive: true,
                                x: 0.1,
                                xanchor: 'left',
                                y: 1.1,
                                yanchor: 'top'
                            }}]
                        }};
                    }}
                    
                    Plotly.newPlot('importancePlot', data, layout);
                </script>
            </div>
            """)
        except Exception as e:
            logger.warning(f"Error creating feature importance plot: {e}")
    
    # 2. Temporal Feature Importance Visualization
    evolution_path = os.path.join(results_dir, "temporal_importance_evolution.csv")
    if os.path.exists(evolution_path):
        try:
            evolution_df = pd.read_csv(evolution_path)
            
            # Prepare data
            windows = evolution_df['window'].tolist()
            top_features = evolution_df['top_feature'].tolist()
            imp_values = evolution_df['importance_value'].tolist()
            
            html_content.append("""
            <div class="plot-container">
                <h3>Temporal Evolution of Feature Importance</h3>
                <div id="evolutionPlot" style="height: 500px;"></div>
                <script>
            """)
            
            html_content.append(f"""
                    var windows = {windows};
                    var topFeatures = {top_features};
                    var impValues = {imp_values};
                    
                    var trace = {{
                        x: windows,
                        y: impValues,
                        mode: 'lines+markers+text',
                        type: 'scatter',
                        text: topFeatures,
                        textposition: 'top center',
                        line: {{
                            color: 'rgb(67, 67, 173)',
                            width: 3
                        }},
                        marker: {{
                            color: 'rgb(67, 67, 173)',
                            size: 12
                        }}
                    }};
                    
                    var layout = {{
                        title: 'Top Feature Evolution During Drift',
                        xaxis: {{
                            title: 'Time Window',
                            zeroline: false
                        }},
                        yaxis: {{
                            title: 'Importance Value',
                            zeroline: false
                        }},
                        shapes: [{{
                            type: 'line',
                            x0: 'Transition',
                            y0: 0,
                            x1: 'Transition',
                            y1: 1,
                            yref: 'paper',
                            line: {{
                                color: 'red',
                                width: 2,
                                dash: 'dash'
                            }}
                        }}],
                        annotations: [{{
                            x: 'Transition',
                            y: 1,
                            yref: 'paper',
                            text: 'Drift Point',
                            showarrow: true,
                            arrowhead: 2,
                            ax: 0,
                            ay: -40
                        }}]
                    }};
                    
                    Plotly.newPlot('evolutionPlot', [trace], layout);
                </script>
            </div>
            """)
        except Exception as e:
            logger.warning(f"Error creating temporal evolution plot: {e}")
    
    # 3. Expert Performance Visualization
    expert_path = os.path.join(results_dir, "expert_drift_impact.csv")
    if os.path.exists(expert_path):
        try:
            expert_df = pd.read_csv(expert_path)
            
            # Prepare data
            specialties = expert_df['specialty'].tolist()
            before_mse = expert_df['before_mse'].tolist()
            after_mse = expert_df['after_mse'].tolist()
            degradation = expert_df['degradation_pct'].tolist()
            
            # Get uncertainty bounds if available
            lower_bounds = expert_df['lower_bound'].tolist() if 'lower_bound' in expert_df.columns else [d * 0.8 for d in degradation]
            upper_bounds = expert_df['upper_bound'].tolist() if 'upper_bound' in expert_df.columns else [d * 1.2 for d in degradation]
            
            # Get severity and recommendations if available
            severities = expert_df['severity'].tolist() if 'severity' in expert_df.columns else ['Unknown'] * len(specialties)
            recommendations = expert_df['recommendation'].tolist() if 'recommendation' in expert_df.columns else ['No recommendation available'] * len(specialties)
            
            # Calculate normalized MSE (as percentage of maximum)
            max_mse = max(max(before_mse), max(after_mse))
            norm_before_mse = [100 * mse / max_mse for mse in before_mse]
            norm_after_mse = [100 * mse / max_mse for mse in after_mse]
            
            # Create log scale versions (adding small epsilon to avoid log(0))
            epsilon = 1e-10
            log_before_mse = [math.log10(mse + epsilon) for mse in before_mse]
            log_after_mse = [math.log10(mse + epsilon) for mse in after_mse]
            
            html_content.append("""
            <div class="plot-container">
                <h3>Expert-Specific Drift Impact</h3>
                <div id="expertPlot" style="height: 500px;"></div>
                <script>
            """)
            
            html_content.append(f"""
                    var specialties = {specialties};
                    var beforeMSE = {before_mse};
                    var afterMSE = {after_mse};
                    var degradation = {degradation};
                    var lowerBounds = {lower_bounds};
                    var upperBounds = {upper_bounds};
                    var severities = {severities};
                    var recommendations = {recommendations};
                    var normBeforeMSE = {norm_before_mse};
                    var normAfterMSE = {norm_after_mse};
                    var logBeforeMSE = {log_before_mse};
                    var logAfterMSE = {log_after_mse};
                    
                    // Raw MSE values
                    var trace1 = {{
                        x: specialties,
                        y: beforeMSE,
                        name: 'Before Drift MSE',
                        type: 'bar',
                        marker: {{color: 'royalblue'}}
                    }};
                    
                    var trace2 = {{
                        x: specialties,
                        y: afterMSE,
                        name: 'After Drift MSE',
                        type: 'bar',
                        marker: {{color: 'firebrick'}}
                    }};
                    
                    // Log scale MSE values
                    var trace3 = {{
                        x: specialties,
                        y: logBeforeMSE,
                        name: 'Before Drift MSE (log)',
                        type: 'bar',
                        marker: {{color: 'royalblue'}},
                        visible: false
                    }};
                    
                    var trace4 = {{
                        x: specialties,
                        y: logAfterMSE,
                        name: 'After Drift MSE (log)',
                        type: 'bar',
                        marker: {{color: 'firebrick'}},
                        visible: false
                    }};
                    
                    // Normalized MSE values
                    var trace5 = {{
                        x: specialties,
                        y: normBeforeMSE,
                        name: 'Before Drift MSE (%)',
                        type: 'bar',
                        marker: {{color: 'royalblue'}},
                        visible: false
                    }};
                    
                    var trace6 = {{
                        x: specialties,
                        y: normAfterMSE,
                        name: 'After Drift MSE (%)',
                        type: 'bar',
                        marker: {{color: 'firebrick'}},
                        visible: false
                    }};
                    
                    // Degradation with clinical impact ranges and uncertainty bounds
                    var trace7 = {{
                        x: specialties,
                        y: degradation,
                        name: 'Degradation (%)',
                        type: 'bar',
                        marker: {{
                            color: degradation,
                            colorscale: [
                                [0, 'green'],
                                [0.3, 'yellow'],
                                [0.6, 'orange'],
                                [1, 'red']
                            ],
                            showscale: true,
                            colorbar: {{
                                title: 'Clinical Impact',
                                tickvals: [0, 30, 60, 100],
                                ticktext: ['Low', 'Moderate', 'High', 'Critical']
                            }}
                        }},
                        error_y: {{
                            type: 'data',
                            symmetric: false,
                            array: upperBounds.map((upper, i) => upper - degradation[i]),
                            arrayminus: degradation.map((deg, i) => deg - lowerBounds[i]),
                            visible: true,
                            color: 'black'
                        }},
                        visible: false
                    }};
                    
                    var data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7];
                    
                    var layout = {{
                        barmode: 'group',
                        title: 'Expert Performance Before vs After Drift',
                        xaxis: {{title: 'Expert Specialty'}},
                        yaxis: {{title: 'Mean Squared Error / Degradation (%)'}},
                        updatemenus: [
                            {{
                                type: 'buttons',
                                direction: 'right',
                                buttons: [
                                    {{
                                        args: [{{visible: [true, true, false, false, false, false, false]}}, 
                                               {{yaxis: {{title: 'Mean Squared Error (Raw)', type: 'linear'}}}}],
                                        label: 'Raw MSE',
                                        method: 'update'
                                    }},
                                    {{
                                        args: [{{visible: [false, false, true, true, false, false, false]}},
                                               {{yaxis: {{title: 'Log10(Mean Squared Error)', type: 'linear'}}}}],
                                        label: 'Log Scale MSE',
                                        method: 'update'
                                    }},
                                    {{
                                        args: [{{visible: [false, false, false, false, true, true, false]}},
                                               {{yaxis: {{title: 'Normalized MSE (%)', type: 'linear'}}}}],
                                        label: 'Normalized MSE',
                                        method: 'update'
                                    }},
                                    {{
                                        args: [{{visible: [false, false, false, false, false, false, true]}},
                                               {{yaxis: {{title: 'Degradation (%)', type: 'linear'}}}}],
                                        label: 'Clinical Impact',
                                        method: 'update'
                                    }}
                                ],
                                pad: {{'r': 10, 't': 10}},
                                showactive: true,
                                x: 0.1,
                                xanchor: 'left',
                                y: 1.2,
                                yanchor: 'top'
                            }},
                            {{
                                type: 'buttons',
                                direction: 'right',
                                buttons: [
                                    {{
                                        args: [{{annotations: [
                                            {{
                                                text: 'Degradation <30%: Low clinical impact',
                                                x: 0.5,
                                                y: 1.05,
                                                xref: 'paper',
                                                yref: 'paper',
                                                showarrow: false,
                                                font: {{
                                                    size: 12,
                                                    color: 'green'
                                                }}
                                            }},
                                            {{
                                                text: '30-60%: Moderate impact, review recommended',
                                                x: 0.5,
                                                y: 1.0,
                                                xref: 'paper',
                                                yref: 'paper',
                                                showarrow: false,
                                                font: {{
                                                    size: 12,
                                                    color: 'orange'
                                                }}
                                            }},
                                            {{
                                                text: '>60%: Critical impact, immediate action required',
                                                x: 0.5,
                                                y: 0.95,
                                                xref: 'paper',
                                                yref: 'paper',
                                                showarrow: false,
                                                font: {{
                                                    size: 12,
                                                    color: 'red'
                                                }}
                                            }}
                                        ]}}],
                                        label: 'Show Clinical Guidance',
                                        method: 'update'
                                    }},
                                    {{
                                        args: [{{annotations: []}}],
                                        label: 'Hide Clinical Guidance',
                                        method: 'update'
                                    }}
                                ],
                                pad: {{'r': 10, 't': 10}},
                                showactive: true,
                                x: 0.6,
                                xanchor: 'left',
                                y: 1.2,
                                yanchor: 'top'
                            }}
                        ]
                    }};
                    
                    Plotly.newPlot('expertPlot', data, layout);
                </script>
            </div>

            <div class="recommendations-container">
                <h3>Expert-Specific Recommendations</h3>
                <table class="recommendations-table" style="width:100%; border-collapse: collapse; margin-top: 20px;">
                    <thead>
                        <tr>
                            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Expert</th>
                            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Severity</th>
                            <th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">Recommendation</th>
                        </tr>
                    </thead>
                    <tbody>
            """ + '\n'.join([f"""
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">{specialty}</td>
                            <td style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: {'#ffdddd' if severity == 'Critical' else '#ffffcc' if severity == 'Moderate' else '#ddffdd'};">{severity}</td>
                            <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">{recommendation}</td>
                        </tr>
            """ for specialty, severity, recommendation in zip(specialties, severities, recommendations)]) + """
                    </tbody>
                </table>
            </div>
            """)
        except Exception as e:
            logger.warning(f"Error creating expert performance plot: {e}")
    
    # Add Explainability Visualizations Section
    html_content.append("""
        <h2>Model Explainability Insights</h2>
        <div class="notification">
            This section provides insights into model behavior using explainers like SHAP and feature importance.
        </div>
    """)
    
    # Check if enhanced validation with explainability is present in results
    has_explainability = False
    explainability_data = {}
    continuous_data = {}
    
    # First, try to find explainability data in the enhanced_validation section
    if isinstance(test_results, dict):
        if 'enhanced_validation' in test_results:
            enhanced_results = test_results.get('enhanced_validation', {})
            # Look for explanation_results key which is used in the MoEValidationEnhancer
            if 'explanation_results' in enhanced_results:
                has_explainability = True
                explainability_data = enhanced_results.get('explanation_results', {})
        # Also look for explainability directly in test_results
        elif 'explanation_results' in test_results:
            has_explainability = True
            explainability_data = test_results.get('explanation_results', {})
    
    # Also check for the separately generated explainability data file
    explainability_file = os.path.join(results_dir, 'explainability_report_data.json')
    if os.path.exists(explainability_file):
        try:
            with open(explainability_file, 'r') as f:
                file_data = json.load(f)
                if 'explanation_results' in file_data:
                    has_explainability = True
                    explainability_data = file_data.get('explanation_results', {})
        except Exception as e:
            logger.warning(f"Error loading explainability data file: {e}")
    
    if has_explainability:
        # Get continuous explainability data if available
        # Check in the explainability results or results directory
        if isinstance(explainability_data, dict):
            # Look for continuous explainability data within explanation results
            if 'continuous_explainability' in explainability_data:
                continuous_data = explainability_data.get('continuous_explainability', {})
            elif 'importance_trends' in explainability_data:
                continuous_data = {'importance_trends': explainability_data.get('importance_trends', {})}
                
        # Also check for continuous explainability data files in the results directory
        continuous_explainability_dir = os.path.join(results_dir, 'continuous_explainability')
        continuous_explanations_file = os.path.join(continuous_explainability_dir, 'continuous_explanations.json')
        
        if os.path.exists(continuous_explanations_file):
            try:
                with open(continuous_explanations_file, 'r') as f:
                    continuous_logs = json.load(f)
                    
                if continuous_logs and isinstance(continuous_logs, list):
                    # Extract feature importance trends from logs
                    trends_data = {}
                    
                    # Process each log entry
                    for entry in continuous_logs:
                        if 'explanations' in entry:
                            for explainer, exp_data in entry['explanations'].items():
                                if 'feature_importance' in exp_data:
                                    fi_data = exp_data['feature_importance']
                                    
                                    if isinstance(fi_data, dict):
                                        for feature, value in fi_data.items():
                                            if feature not in trends_data:
                                                trends_data[feature] = []
                                            trends_data[feature].append(value)
                    
                    if trends_data:
                        continuous_data['importance_trends'] = trends_data
                        has_explainability = True
            except Exception as e:
                logger.warning(f"Error loading continuous explanations: {e}")
        
        # Feature Importance Visualization Section
        html_content.append("""
            <div class="plot-container">
                <h3>Feature Importance Analysis</h3>
                <div id="featureImportancePlot" style="height: 500px;"></div>
                <script>
        """)
        
        # Extract and process feature importance data
        feature_names = []
        importance_values = []
        explainer_types = []
        
        # Process explainability data from different explainers
        for explainer_type, data in explainability_data.items():
            if isinstance(data, dict) and 'feature_importance' in data:
                fi_data = data['feature_importance']
                
                if isinstance(fi_data, dict):
                    # For dictionary format (feature_name: importance_value)
                    # Handle complex data types for feature importance
                    sorted_items = []
                    for name, value in fi_data.items():
                        try:
                            # Try to convert to float if it's a number
                            if isinstance(value, (int, float, str)):
                                sorted_items.append((name, float(value)))
                            else:
                                # If it's a complex structure like a dict or array,
                                # try to extract a representative value or skip
                                if isinstance(value, dict) and value:
                                    # Use the first value in the dictionary
                                    first_key = next(iter(value))
                                    if isinstance(value[first_key], (int, float, str)):
                                        sorted_items.append((name, float(value[first_key])))
                                elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                                    # Use the mean or first value
                                    if isinstance(value[0], (int, float, str)):
                                        sorted_items.append((name, float(value[0])))
                        except (ValueError, TypeError):
                            # Skip values that can't be converted to float
                            pass
                    
                    # Sort and get top 10
                    sorted_items = sorted(sorted_items, key=lambda x: abs(x[1]), reverse=True)[:10]
                    for name, value in sorted_items:
                        feature_names.append(str(name))
                        try:
                            importance_values.append(float(value))
                        except (ValueError, TypeError):
                            importance_values.append(0.0)
                        explainer_types.append(explainer_type)
                        
                elif isinstance(fi_data, (list, np.ndarray)):
                    # For list/array format with unnamed features
                    for i, value in enumerate(fi_data[:10]):  # Top 10 features
                        feature_names.append(f"Feature {i}")
                        try:
                            importance_values.append(float(value))
                        except (ValueError, TypeError):
                            importance_values.append(0.0)
                        explainer_types.append(explainer_type)
        
        if feature_names:
            # Create Plotly visualization for feature importance
            html_content.append(f"""
                    var feature_names = {feature_names};
                    var importance_values = {importance_values};
                    var explainer_types = {explainer_types};
                    
                    var trace = {{
                        x: importance_values,
                        y: feature_names,
                        type: 'bar',
                        orientation: 'h',
                        marker: {{
                            color: importance_values.map(function(val) {{
                                return val >= 0 ? 'rgba(55, 128, 191, 0.7)' : 'rgba(219, 64, 82, 0.7)';
                            }})
                        }}
                    }};
                    
                    var layout = {{
                        title: 'Top Feature Importance',
                        xaxis: {{
                            title: 'Importance Value'
                        }},
                        yaxis: {{
                            title: 'Feature',
                            automargin: true
                        }},
                        margin: {{
                            l: 150,
                            r: 50,
                            b: 50,
                            t: 50,
                            pad: 4
                        }}
                    }};
                    
                    Plotly.newPlot('featureImportancePlot', [trace], layout);
                </script>
            </div>
            """)
        else:
            html_content.append("""
            <div class="plot-container">
                <p>No feature importance data available for visualization.</p>
            </div>
            """)
        
        # Feature Importance Trends Section (if available)
        if 'importance_trends' in continuous_data:
            html_content.append("""
            <div class="plot-container">
                <h3>Feature Importance Trends Over Time</h3>
                <div id="importanceTrendsPlot" style="height: 500px;"></div>
                <script>
            """)
            
            trends_data = continuous_data['importance_trends']
            if isinstance(trends_data, dict) and trends_data:
                # Get top features by average importance
                top_features = []
                top_values = []
                
                for feature, values in trends_data.items():
                    if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                        avg_importance = np.mean([abs(float(v)) for v in values])
                        top_features.append((feature, values, avg_importance))
                
                # Sort and get top 5 features
                top_features.sort(key=lambda x: x[2], reverse=True)
                top_features = top_features[:5]  # Limit to top 5
                
                if top_features:
                    feature_names = [f[0] for f in top_features]
                    all_values = []
                    max_length = max([len(f[1]) for f in top_features])
                    
                    for _, values, _ in top_features:
                        # Convert to list of floats and pad if necessary
                        float_values = [float(v) for v in values]
                        if len(float_values) < max_length:
                            float_values.extend([None] * (max_length - len(float_values)))
                        all_values.append(float_values)
                    
                    # X-axis time points
                    time_points = list(range(1, max_length + 1))
                    
                    html_content.append(f"""
                        var feature_names = {feature_names};
                        var time_points = {time_points};
                        var all_values = {all_values};
                        
                        var data = [];
                        
                        for (var i = 0; i < feature_names.length; i++) {{
                            var trace = {{
                                x: time_points,
                                y: all_values[i],
                                type: 'scatter',
                                mode: 'lines+markers',
                                name: feature_names[i]
                            }};
                            data.push(trace);
                        }}
                        
                        var layout = {{
                            title: 'Feature Importance Trends Over Time',
                            xaxis: {{
                                title: 'Time Point'
                            }},
                            yaxis: {{
                                title: 'Importance Value'
                            }}
                        }};
                        
                        Plotly.newPlot('importanceTrendsPlot', data, layout);
                    </script>
                </div>
                """)
                else:
                    html_content.append("""
                    <p>No trend data available for visualization.</p>
                    </script>
                </div>
                """)
            else:
                html_content.append("""
                <p>No feature importance trends data available.</p>
                </script>
                </div>
                """)
        
        # Look for explainability images in the results directory
        explainability_dir = os.path.join(results_dir, 'continuous_explainability')
        if os.path.exists(explainability_dir):
            html_content.append("""
            <div class="plot-container">
                <h3>Explainability Visualizations</h3>
            """)
            
            # Find PNG images from explainers
            png_files = [f for f in os.listdir(explainability_dir) if f.endswith('.png') and 'explanation' in f]
            if png_files:
                # Sort by timestamp (assuming filenames contain timestamps)
                png_files.sort(reverse=True)
                
                # Format as a grid of images with captions
                html_content.append('<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px;">')
                
                # Display the most recent 6 visualizations
                for i, png_file in enumerate(png_files[:6]):
                    file_path = os.path.join('continuous_explainability', png_file)
                    rel_path = os.path.relpath(os.path.join(explainability_dir, png_file), os.path.dirname(report_path))
                    
                    # Clean up filename for display
                    display_name = png_file.replace('explanation_', '').replace('.png', '').replace('_', ' ')
                    
                    html_content.append(f'''
                    <div style="border: 1px solid #eee; padding: 10px; border-radius: 5px;">
                        <h4>{display_name}</h4>
                        <img src="{rel_path}" style="width: 100%; max-height: 300px; object-fit: contain;">
                    </div>
                    ''')
                
                html_content.append('</div>')
                
                # Add a note about additional visualizations
                if len(png_files) > 6:
                    html_content.append(f'<p style="margin-top: 20px;">{len(png_files) - 6} more visualizations available in the explainability directory.</p>')
            else:
                html_content.append('<p>No explainability visualizations found.</p>')
            
            html_content.append('</div>')
    else:
        html_content.append("""
        <div class="notification warning">
            No explainability data available. Run with the --enable-continuous-explain flag to generate explainability insights.
        </div>
        """)

    # Close HTML
    html_content.append("""
        </div>
    </body>
    </html>
    """)
    
    # Write HTML file
    with open(report_path, 'w') as f:
        f.write('\n'.join(html_content))
    
    logger.info(f"Interactive HTML report generated at {report_path}")
    return str(report_path)

class AutomaticDriftNotifier:
    """Class to handle drift notifications based on validation results."""
    
    def __init__(self, threshold=0.5, notification_method='file'):
        """Initialize the notifier with threshold and notification method.
        
        Args:
            threshold: Threshold for drift magnitude (normalized 0-1)
            notification_method: How to notify (file, email, etc.)
        """
        self.threshold = threshold
        self.notification_method = notification_method
        
    def check_drift_severity(self, test_results):
        """Check drift severity from test results.
        
        Args:
            test_results: Dictionary of test results
            
        Returns:
            Tuple (severity_level, message)
        """
        # Default values
        severity = "low"
        message = "No significant drift detected."
        
        # Check if drift detected
        if 'drift_detected' in test_results and test_results['drift_detected'].get('passed', True):
            details = test_results['drift_detected'].get('details', "")
            
            # Extract drift magnitude if available
            try:
                import re
                magnitude_match = re.search(r'Magnitude: ([\d.]+)', details)
                if magnitude_match:
                    magnitude = float(magnitude_match.group(1))
                    
                    # Normalize magnitude (assuming maximum around 1,000,000)
                    norm_magnitude = min(magnitude / 1000000, 1.0)
                    
                    # Determine severity
                    if norm_magnitude > 0.8:
                        severity = "critical"
                        message = f"CRITICAL drift detected with magnitude {magnitude:.2f}. Immediate attention required."
                    elif norm_magnitude > 0.5:
                        severity = "high"
                        message = f"HIGH severity drift detected with magnitude {magnitude:.2f}. Review recommended."
                    elif norm_magnitude > 0.2:
                        severity = "medium"
                        message = f"MEDIUM severity drift detected with magnitude {magnitude:.2f}. Monitor carefully."
                    else:
                        severity = "low"
                        message = f"LOW severity drift detected with magnitude {magnitude:.2f}. No immediate action required."
            except Exception as e:
                logger.warning(f"Error parsing drift magnitude: {e}")
                severity = "unknown"
                message = "Drift detected but severity could not be determined."
        
        return severity, message
    
    def send_notification(self, test_results, results_dir):
        """Send notification based on drift severity.
        
        Args:
            test_results: Dictionary of test results
            results_dir: Directory for storing notification files
            
        Returns:
            Path to notification file or message about notification sent
        """
        severity, message = self.check_drift_severity(test_results)
        
        # Only notify if severity is above threshold
        severity_levels = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0, "unknown": 0.5}
        if severity_levels.get(severity, 0) < self.threshold:
            logger.info(f"Drift severity {severity} below threshold. No notification sent.")
            return None
            
        if self.notification_method == 'file':
            # Create notification file
            results_dir = str(results_dir) if not isinstance(results_dir, str) else results_dir
            notif_dir = os.path.join(results_dir, "notifications")
            Path(notif_dir).mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            notif_path = os.path.join(notif_dir, f"drift_notification_{severity}_{timestamp}.txt")
            
            with open(notif_path, 'w') as f:
                f.write(f"DRIFT NOTIFICATION - {severity.upper()} SEVERITY\n")
                f.write(f"Timestamp: {timestamp.replace('_', ' ')}\n\n")
                f.write(f"{message}\n\n")
                
                # Add test results
                f.write("Test Results Summary:\n")
                for test_name, result in test_results.items():
                    status = "Passed" if result.get('passed', False) else "Failed"
                    f.write(f"- {test_name}: {status}\n")
                    if 'details' in result:
                        f.write(f"  Details: {result['details']}\n")
                
                # Add recommendation based on severity
                f.write("\nRECOMMENDED ACTIONS:\n")
                if severity == "critical":
                    f.write("1. Immediately pause model in production\n")
                    f.write("2. Analyze drift cause and impact\n")
                    f.write("3. Retrain model with new data\n")
                    f.write("4. Perform full validation before re-deploying\n")
                elif severity == "high":
                    f.write("1. Review model performance in production\n")
                    f.write("2. Schedule model retraining\n")
                    f.write("3. Increase monitoring frequency\n")
                elif severity == "medium":
                    f.write("1. Increase monitoring frequency\n")
                    f.write("2. Prepare data for potential retraining\n")
                else:
                    f.write("1. Continue regular monitoring\n")
            
            logger.info(f"Drift notification saved to {notif_path}")
            return notif_path
            
        elif self.notification_method == 'email':
            # Placeholder for email notification
            logger.info(f"Would send email with severity {severity}: {message}")
            return f"Email notification would be sent with severity {severity}"
            
        return None
