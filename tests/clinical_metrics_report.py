"""
Clinical Metrics Reporting Module for MoE Validation Framework

This module provides functions to generate clinical performance metrics visualizations.
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

def generate_clinical_metrics_section(test_results, results_dir):
    """Generate visualizations for clinical metrics.
    
    Args:
        test_results: Dictionary of test results
        results_dir: Directory with result files
        
    Returns:
        List of HTML content for the clinical metrics section
    """
    logger.info("Generating clinical metrics section...")
    
    html_content = []
    
    # Start section
    html_content.append("""
    <div class="section">
        <h2>Clinical Performance Metrics</h2>
        <div class="description">
            <p>This section displays clinical performance metrics for the MoE model, including 
            MSE degradation over time, severity-adjusted metrics, and clinical utility scores.</p>
        </div>
    """)
    
    # Generate MSE degradation over time chart
    html_content.append("""
    <div class="subsection">
        <h3>MSE Degradation Over Time</h3>
        <div class="chart-container">
            <div id="mse-degradation-chart"></div>
        </div>
        <div class="description">
            <p>This chart shows how Mean Squared Error (MSE) changes over time, 
            indicating potential model degradation during different drift scenarios.</p>
        </div>
    </div>
    """)
    
    # Generate severity-adjusted metrics
    html_content.append("""
    <div class="subsection">
        <h3>Severity-Adjusted Performance Metrics</h3>
        <div class="chart-container">
            <div id="severity-metrics-chart"></div>
        </div>
        <div class="description">
            <p>Performance metrics adjusted by clinical severity, weighting errors 
            based on their potential clinical impact.</p>
        </div>
    </div>
    """)
    
    # Generate utility metrics
    html_content.append("""
    <div class="subsection">
        <h3>Clinical Utility Composite Score</h3>
        <div class="chart-container">
            <div id="utility-metrics-chart"></div>
        </div>
        <div class="description">
            <p>Composite metrics combining prediction accuracy and clinical importance 
            to provide a holistic view of model utility in clinical settings.</p>
        </div>
    </div>
    """)
    
    # Get sample data for visualizations
    # In a real implementation, this would come from test_results
    # For now, we'll generate sample data
    
    # MSE degradation over time data
    timestamps = np.linspace(0, 100, 50)
    base_mse = np.concatenate([
        np.random.normal(0.05, 0.01, 20),
        np.random.normal(0.15, 0.03, 10),  # Drift period
        np.random.normal(0.08, 0.02, 20)   # Recovery
    ])
    
    # Add Plotly.js code for charts
    html_content.append("""
    <script>
    // MSE Degradation Chart
    (function() {
        var timestamps = %s;
        var mse_values = %s;
        
        var trace = {
            x: timestamps,
            y: mse_values,
            mode: 'lines+markers',
            name: 'MSE',
            line: {
                color: 'rgb(55, 83, 176)',
                width: 2
            },
            marker: {
                size: 6,
                color: 'rgb(55, 83, 176)',
                line: {
                    color: 'white',
                    width: 0.5
                }
            }
        };
        
        var layout = {
            title: 'MSE Degradation Over Time',
            xaxis: {
                title: 'Time',
                showgrid: true,
                zeroline: true
            },
            yaxis: {
                title: 'Mean Squared Error',
                showgrid: true,
                zeroline: true
            },
            shapes: [{
                type: 'rect',
                xref: 'x',
                yref: 'paper',
                x0: 20,
                y0: 0,
                x1: 30,
                y1: 1,
                fillcolor: 'rgba(255, 0, 0, 0.1)',
                line: {
                    width: 0
                }
            }],
            annotations: [{
                x: 25,
                y: 0.18,
                xref: 'x',
                yref: 'y',
                text: 'Drift Period',
                showarrow: true,
                arrowhead: 2,
                ax: 0,
                ay: -40
            }],
            showlegend: false,
            margin: {l: 40, r: 20, t: 60, b: 40}
        };
        
        Plotly.newPlot('mse-degradation-chart', [trace], layout, {responsive: true});
    })();
    
    // Severity-Adjusted Metrics Chart
    (function() {
        var categories = ['Low', 'Medium', 'High', 'Critical'];
        var standard_metrics = [0.92, 0.85, 0.78, 0.70];
        var severity_adjusted = [0.95, 0.89, 0.82, 0.65];
        
        var trace1 = {
            x: categories,
            y: standard_metrics,
            name: 'Standard Metrics',
            type: 'bar',
            marker: {
                color: 'rgb(55, 83, 176)',
                opacity: 0.7
            }
        };
        
        var trace2 = {
            x: categories,
            y: severity_adjusted,
            name: 'Severity-Adjusted',
            type: 'bar',
            marker: {
                color: 'rgb(26, 118, 255)',
                opacity: 0.7
            }
        };
        
        var layout = {
            title: 'Performance by Clinical Severity',
            xaxis: {
                title: 'Clinical Severity Level'
            },
            yaxis: {
                title: 'Performance Score (higher is better)',
                range: [0, 1]
            },
            barmode: 'group',
            bargap: 0.15,
            bargroupgap: 0.1,
            margin: {l: 40, r: 20, t: 60, b: 40}
        };
        
        Plotly.newPlot('severity-metrics-chart', [trace1, trace2], layout, {responsive: true});
    })();
    
    // Clinical Utility Composite Score Chart
    (function() {
        var data = [{
            type: 'scatterpolar',
            r: [0.8, 0.7, 0.9, 0.65, 0.85],
            theta: ['Accuracy', 'Clinical Impact', 'Timeliness', 'Explainability', 'Patient Relevance'],
            fill: 'toself',
            name: 'Current Model',
            line: {
                color: 'rgb(55, 83, 176)'
            }
        }, {
            type: 'scatterpolar',
            r: [0.6, 0.8, 0.7, 0.8, 0.7],
            theta: ['Accuracy', 'Clinical Impact', 'Timeliness', 'Explainability', 'Patient Relevance'],
            fill: 'toself',
            name: 'Baseline Model',
            line: {
                color: 'rgb(126, 126, 126)'
            }
        }];
        
        var layout = {
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 1]
                }
            },
            title: 'Clinical Utility Composite Score',
            showlegend: true,
            margin: {l: 40, r: 40, t: 60, b: 40}
        };
        
        Plotly.newPlot('utility-metrics-chart', data, layout, {responsive: true});
    })();
    </script>
    """ % (timestamps.tolist(), base_mse.tolist()))
    
    # Close the section
    html_content.append('</div>')
    
    return html_content
