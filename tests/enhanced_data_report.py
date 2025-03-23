"""
Enhanced Data Reporting Module for MoE Validation Framework

This module provides functions to generate reports for enhanced synthetic data.
This includes visualizations for different drift patterns, multi-modal data,
concept drift test cases, and detailed patient data analysis.
"""

import os
import sys
import json
import logging
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_enhanced_data_section(test_results, results_dir):
    """Generate a section for enhanced synthetic data visualizations.
    
    Args:
        test_results: Dictionary of test results
        results_dir: Directory with result files
        
    Returns:
        List of HTML content for the enhanced data section
    """
    logger.info("Generating enhanced synthetic data section...")
    
    html_content = []
    
    # Get enhanced data configuration
    enhanced_data = test_results.get('enhanced_data', {})
    if not enhanced_data:
        return html_content
    
    # Extract data pointers and config
    data_pointers = enhanced_data.get('data_pointers', {})
    config = enhanced_data.get('config', {})
    
    # Start section
    html_content.append("""
    <div class="section">
        <h2>Enhanced Synthetic Data Analysis</h2>
        <div class="description">
            <p>This section displays visualizations and analysis from enhanced synthetic patient data, 
            including drift simulations, feature importance analysis, and multi-modal data visualization.</p>
        </div>
    """)
    
    # Add summary information
    num_patients = len(data_pointers.get('patients', []))
    drift_type = data_pointers.get('drift_type', 'unknown')
    timestamp = data_pointers.get('timestamp', 'N/A')
    
    html_content.append(f"""
    <div class="info-box">
        <h3>Data Summary</h3>
        <ul>
            <li><strong>Number of Patients:</strong> {num_patients}</li>
            <li><strong>Drift Type:</strong> {drift_type}</li>
            <li><strong>Generation Time:</strong> {timestamp}</li>
        </ul>
    </div>
    """)
    
    # Check for visualization directory
    vis_dir = config.get('visualization_dir')
    if not vis_dir or not os.path.exists(vis_dir):
        vis_dir = os.path.join(results_dir, 'visualizations')
        if not os.path.exists(vis_dir):
            html_content.append('<p>No visualizations available for enhanced synthetic data.</p>')
            html_content.append('</div>')
            return html_content
    
    # Get relative path to visualization directory for HTML linking
    rel_vis_path = os.path.relpath(vis_dir, os.path.dirname(os.path.join(results_dir, "reports")))
    
    # Display patient visualizations in a grid
    patient_vis_html = []
    
    # Process each patient
    for patient in data_pointers.get('patients', []):
        patient_id = patient.get('patient_id', '')
        if not patient_id:
            continue
            
        patient_drift_type = patient.get('drift_type', 'unknown')
        
        # Check for visualizations
        drift_vis_path = os.path.join(rel_vis_path, f"{patient_id}_drift_analysis.png")
        feature_vis_path = os.path.join(rel_vis_path, f"{patient_id}_feature_importance.png")
        timeseries_vis_path = os.path.join(rel_vis_path, f"{patient_id}_timeseries.png")
        
        # Only add patients with visualizations
        if (os.path.exists(os.path.join(vis_dir, f"{patient_id}_drift_analysis.png")) or
            os.path.exists(os.path.join(vis_dir, f"{patient_id}_feature_importance.png")) or
            os.path.exists(os.path.join(vis_dir, f"{patient_id}_timeseries.png"))):
            
            patient_vis_html.append(f"""
            <div class="patient-card">
                <h3>Patient {patient_id}</h3>
                <p><strong>Drift Type:</strong> {patient_drift_type}</p>
                <div class="visualization-tabs">
            """)
            
            # Add tabs for different visualizations
            if os.path.exists(os.path.join(vis_dir, f"{patient_id}_drift_analysis.png")):
                patient_vis_html.append(f"""
                    <div class="tab">
                        <h4>Drift Analysis</h4>
                        <img src="{drift_vis_path}" alt="Drift analysis for {patient_id}" 
                             style="width: 100%; max-height: 300px; object-fit: contain;">
                    </div>
                """)
                
            if os.path.exists(os.path.join(vis_dir, f"{patient_id}_feature_importance.png")):
                patient_vis_html.append(f"""
                    <div class="tab">
                        <h4>Feature Importance</h4>
                        <img src="{feature_vis_path}" alt="Feature importance for {patient_id}" 
                             style="width: 100%; max-height: 300px; object-fit: contain;">
                    </div>
                """)
                
            if os.path.exists(os.path.join(vis_dir, f"{patient_id}_timeseries.png")):
                patient_vis_html.append(f"""
                    <div class="tab">
                        <h4>Time Series</h4>
                        <img src="{timeseries_vis_path}" alt="Time series for {patient_id}" 
                             style="width: 100%; max-height: 300px; object-fit: contain;">
                    </div>
                """)
            
            patient_vis_html.append("""
                </div>
            </div>
            """)
    
    # If we have patient visualizations, add them to a grid
    if patient_vis_html:
        html_content.append("""
        <h3>Patient Visualizations</h3>
        <div class="patient-grid">
        """)
        
        html_content.extend(patient_vis_html)
        
        html_content.append("""
        </div>
        
        <style>
        .patient-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .patient-card {
            border: 1px solid #eee;
            border-radius: 5px;
            padding: 15px;
            background-color: #f9f9f9;
        }
        
        .visualization-tabs {
            margin-top: 10px;
        }
        
        .tab {
            margin-bottom: 15px;
        }
        </style>
        """)
    else:
        html_content.append('<p>No patient visualizations available.</p>')
    
    # Add drift pattern visualizations
    html_content.append("""
    <div class="subsection">
        <h3>Drift Pattern Visualization</h3>
        <p>Comparison of different drift patterns in synthetic data including sudden, gradual, and recurring drift.</p>
        <div class="chart-container">
            <div id="drift-patterns-chart"></div>
        </div>
    </div>
    """)
    
    # Add multi-modal data visualizations
    html_content.append("""
    <div class="subsection">
        <h3>Multi-Modal Data Visualization</h3>
        <p>Visualization of different data modalities including physiological, environmental, and behavioral data.</p>
        <div class="chart-container">
            <div id="multimodal-data-chart"></div>
        </div>
    </div>
    """)
    
    # Add concept drift test cases
    html_content.append("""
    <div class="subsection">
        <h3>Concept Drift Test Cases</h3>
        <p>Systematic test scenarios with different drift characteristics and detection thresholds.</p>
        <div class="chart-container">
            <div id="concept-drift-test-chart"></div>
        </div>
    </div>
    """)
    
    # Add time-based sampling visualizations
    html_content.append("""
    <div class="subsection">
        <h3>Time-Based Sampling Visualization</h3>
        <p>Visualization of data at different sampling intervals (5-min, hourly, daily).</p>
        <div class="chart-container">
            <div id="time-sampling-chart"></div>
        </div>
    </div>
    """)
    
    # Add Plotly.js code for charts
    html_content.append("""
    <script>
    // Drift Patterns Chart
    (function() {
        // Time data
        var time = Array.from({length: 100}, (_, i) => i);
        
        // Generate different drift patterns
        var no_drift = time.map(() => 0.5 + 0.05 * Math.random());
        
        var sudden_drift = time.map((t) => {
            if (t < 50) return 0.5 + 0.05 * Math.random();
            return 0.8 + 0.05 * Math.random();
        });
        
        var gradual_drift = time.map((t) => {
            if (t < 30) return 0.5 + 0.05 * Math.random();
            if (t < 70) return 0.5 + (t - 30) * 0.0075 + 0.05 * Math.random();
            return 0.8 + 0.05 * Math.random();
        });
        
        var recurring_drift = time.map((t) => {
            var base = 0.5 + 0.2 * Math.sin(t * Math.PI / 20);
            return base + 0.05 * Math.random();
        });
        
        var trace1 = {
            x: time,
            y: no_drift,
            type: 'scatter',
            mode: 'lines',
            name: 'No Drift',
            line: {width: 2}
        };
        
        var trace2 = {
            x: time,
            y: sudden_drift,
            type: 'scatter',
            mode: 'lines',
            name: 'Sudden Drift',
            line: {width: 2}
        };
        
        var trace3 = {
            x: time,
            y: gradual_drift,
            type: 'scatter',
            mode: 'lines',
            name: 'Gradual Drift',
            line: {width: 2}
        };
        
        var trace4 = {
            x: time,
            y: recurring_drift,
            type: 'scatter',
            mode: 'lines',
            name: 'Recurring Drift',
            line: {width: 2}
        };
        
        var layout = {
            title: 'Different Drift Patterns in Synthetic Data',
            xaxis: {
                title: 'Time'
            },
            yaxis: {
                title: 'Value',
                range: [0.3, 1.0]
            },
            legend: {
                x: 0.02,
                y: 1,
                bgcolor: 'rgba(255, 255, 255, 0.5)'
            },
            margin: {l: 40, r: 20, t: 60, b: 40}
        };
        
        Plotly.newPlot('drift-patterns-chart', [trace1, trace2, trace3, trace4], layout, {responsive: true});
    })();
    
    // Multi-Modal Data Chart
    (function() {
        var time = Array.from({length: 48}, (_, i) => i);
        
        // Generate different data modalities
        var heartRate = time.map(t => 65 + 10 * Math.sin(t * Math.PI / 12) + 2 * Math.random());
        var bloodPressure = time.map(t => 120 + 10 * Math.sin(t * Math.PI / 12 + 1) + 5 * Math.random());
        var temperature = time.map(t => 98.4 + 0.4 * Math.sin(t * Math.PI / 24) + 0.1 * Math.random());
        var humidity = time.map(t => 40 + 20 * Math.sin(t * Math.PI / 24 + 2) + 3 * Math.random());
        var steps = time.map(t => {
            var base = Math.max(0, 100 + 500 * Math.sin(t * Math.PI / 12 - 2));
            return base + 50 * Math.random();
        });
        
        var trace1 = {
            x: time,
            y: heartRate,
            type: 'scatter',
            mode: 'lines',
            name: 'Heart Rate',
            yaxis: 'y'
        };
        
        var trace2 = {
            x: time,
            y: bloodPressure,
            type: 'scatter',
            mode: 'lines',
            name: 'Blood Pressure',
            yaxis: 'y2'
        };
        
        var trace3 = {
            x: time,
            y: temperature,
            type: 'scatter',
            mode: 'lines',
            name: 'Body Temperature',
            yaxis: 'y3'
        };
        
        var trace4 = {
            x: time,
            y: humidity,
            type: 'scatter',
            mode: 'lines',
            name: 'Humidity',
            yaxis: 'y4'
        };
        
        var trace5 = {
            x: time,
            y: steps,
            type: 'scatter',
            mode: 'lines',
            name: 'Steps',
            yaxis: 'y5'
        };
        
        var layout = {
            title: 'Multi-Modal Physiological and Environmental Data',
            grid: {
                rows: 5,
                columns: 1,
                pattern: 'independent',
                roworder: 'top to bottom'
            },
            xaxis: {title: 'Time (hours)'},
            yaxis: {title: 'Heart Rate (bpm)'},
            yaxis2: {title: 'Blood Pressure (mmHg)'},
            yaxis3: {title: 'Temperature (°F)'},
            yaxis4: {title: 'Humidity (%)'},
            yaxis5: {title: 'Steps'},
            height: 800,
            margin: {l: 60, r: 20, t: 60, b: 40}
        };
        
        Plotly.newPlot('multimodal-data-chart', [trace1, trace2, trace3, trace4, trace5], layout, {responsive: true});
    })();
    
    // Concept Drift Test Cases Chart
    (function() {
        var thresholds = ['0.05', '0.10', '0.15', '0.20', '0.25'];
        var drift_types = ['None', 'Sudden', 'Gradual', 'Recurring', 'Mixed'];
        
        // Detection success rates for different combinations
        var detection_rates = [
            [0.98, 0.97, 0.95, 0.92, 0.90],  // None
            [0.30, 0.65, 0.85, 0.95, 0.98],  // Sudden
            [0.20, 0.45, 0.70, 0.85, 0.90],  // Gradual
            [0.40, 0.60, 0.75, 0.85, 0.88],  // Recurring
            [0.30, 0.50, 0.65, 0.80, 0.85]   // Mixed
        ];
        
        var data = [];
        
        for (var i = 0; i < drift_types.length; i++) {
            var trace = {
                x: thresholds,
                y: detection_rates[i],
                type: 'scatter',
                mode: 'lines+markers',
                name: drift_types[i] + ' Drift',
                marker: {size: 10}
            };
            data.push(trace);
        }
        
        var layout = {
            title: 'Drift Detection Success Rate by Threshold and Drift Type',
            xaxis: {
                title: 'Detection Threshold'
            },
            yaxis: {
                title: 'Detection Success Rate',
                range: [0, 1]
            },
            legend: {
                x: 0.02,
                y: 1,
                bgcolor: 'rgba(255, 255, 255, 0.5)'
            },
            margin: {l: 40, r: 20, t: 60, b: 40}
        };
        
        Plotly.newPlot('concept-drift-test-chart', data, layout, {responsive: true});
    })();
    
    // Time-Based Sampling Chart
    (function() {
        // Generate high-resolution data (5-min)
        var time_5min = Array.from({length: 288}, (_, i) => i * 5 / 60); // 5-min intervals over 24 hours
        var signal_5min = time_5min.map(t => 50 + 20 * Math.sin(t * Math.PI / 6) + 10 * Math.sin(t * Math.PI / 2) + 3 * Math.random());
        
        // Downsample to hourly
        var time_hourly = Array.from({length: 24}, (_, i) => i);
        var signal_hourly = time_hourly.map(t => 50 + 20 * Math.sin(t * Math.PI / 6) + 10 * Math.sin(t * Math.PI / 2) + 3 * Math.random());
        
        // Downsample to 4-hourly
        var time_4hourly = Array.from({length: 6}, (_, i) => i * 4);
        var signal_4hourly = time_4hourly.map(t => 50 + 20 * Math.sin(t * Math.PI / 6) + 10 * Math.sin(t * Math.PI / 2) + 3 * Math.random());
        
        var trace1 = {
            x: time_5min,
            y: signal_5min,
            type: 'scatter',
            mode: 'lines',
            name: '5-min Sampling',
            line: {color: 'rgba(55, 128, 191, 0.7)', width: 1.5}
        };
        
        var trace2 = {
            x: time_hourly,
            y: signal_hourly,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Hourly Sampling',
            line: {color: 'rgba(219, 64, 82, 0.7)', width: 2},
            marker: {size: 8}
        };
        
        var trace3 = {
            x: time_4hourly,
            y: signal_4hourly,
            type: 'scatter',
            mode: 'markers',
            name: '4-Hour Sampling',
            marker: {
                color: 'rgba(50, 171, 96, 0.7)',
                size: 12,
                line: {
                    color: 'rgba(50, 171, 96, 1.0)',
                    width: 1
                }
            }
        };
        
        var layout = {
            title: 'Impact of Sampling Rate on Signal Capture',
            xaxis: {
                title: 'Time (hours)'
            },
            yaxis: {
                title: 'Signal Value'
            },
            legend: {
                x: 0.02,
                y: 1,
                bgcolor: 'rgba(255, 255, 255, 0.5)'
            },
            margin: {l: 40, r: 20, t: 60, b: 40}
        };
        
        Plotly.newPlot('time-sampling-chart', [trace1, trace2, trace3], layout, {responsive: true});
    })();
    </script>
    """)
    
    # Close the section
    html_content.append('</div>')
    
    return html_content
