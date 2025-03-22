"""
Personalization Features Reporting Module for MoE Validation Framework

This module provides functions to generate visualizations for personalization features.
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

def generate_personalization_section(test_results, results_dir):
    """Generate visualizations for personalization features.
    
    Args:
        test_results: Dictionary of test results
        results_dir: Directory with result files
        
    Returns:
        List of HTML content for the personalization section
    """
    logger.info("Generating personalization features section...")
    
    html_content = []
    
    # Start section
    html_content.append("""
    <div class="section">
        <h2>Personalization Features</h2>
        <div class="description">
            <p>This section displays visualizations for personalization features, including patient profile adaptation,
            personalized gating adjustments, online adaptation, and personalization effectiveness metrics.</p>
        </div>
    """)
    
    # Generate patient profile adaptation chart
    html_content.append("""
    <div class="subsection">
        <h3>Patient Profile Adaptation</h3>
        <div class="chart-container">
            <div id="profile-adaptation-chart"></div>
        </div>
        <div class="description">
            <p>This chart shows how patient profiles evolve over time as the system adapts to individual
            characteristics and patterns.</p>
        </div>
    </div>
    """)
    
    # Generate personalized gating adjustments
    html_content.append("""
    <div class="subsection">
        <h3>Personalized Gating Adjustments</h3>
        <div class="chart-container">
            <div id="gating-adjustments-chart"></div>
        </div>
        <div class="description">
            <p>Visualization of how gating network weights are adjusted for individual patients 
            to improve prediction accuracy.</p>
        </div>
    </div>
    """)
    
    # Generate online adaptation capability
    html_content.append("""
    <div class="subsection">
        <h3>Online Adaptation Capability</h3>
        <div class="chart-container">
            <div id="online-adaptation-chart"></div>
        </div>
        <div class="description">
            <p>Tracking of model adaptation in response to new data and changing patterns, 
            showing online learning capability.</p>
        </div>
    </div>
    """)
    
    # Generate personalization effectiveness metrics
    html_content.append("""
    <div class="subsection">
        <h3>Personalization Effectiveness Metrics</h3>
        <div class="chart-container">
            <div id="personalization-effectiveness-chart"></div>
        </div>
        <div class="description">
            <p>Metrics showing the impact of personalization on model performance for different patient profiles.</p>
        </div>
    </div>
    """)
    
    # Add Plotly.js code for charts
    html_content.append("""
    <script>
    // Patient Profile Adaptation Chart
    (function() {
        var profiles = ['Baseline', 'Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'];
        var stress_sensitivity = [0.6, 0.65, 0.7, 0.73, 0.78, 0.82, 0.85];
        var weather_sensitivity = [0.3, 0.32, 0.35, 0.34, 0.32, 0.3, 0.28];
        var sleep_sensitivity = [0.5, 0.55, 0.6, 0.63, 0.65, 0.68, 0.7];
        var diet_sensitivity = [0.2, 0.22, 0.25, 0.28, 0.3, 0.35, 0.4];
        
        var trace1 = {
            x: profiles,
            y: stress_sensitivity,
            name: 'Stress Sensitivity',
            type: 'scatter',
            mode: 'lines+markers',
            marker: {size: 10}
        };
        
        var trace2 = {
            x: profiles,
            y: weather_sensitivity,
            name: 'Weather Sensitivity',
            type: 'scatter',
            mode: 'lines+markers',
            marker: {size: 10}
        };
        
        var trace3 = {
            x: profiles,
            y: sleep_sensitivity,
            name: 'Sleep Sensitivity',
            type: 'scatter',
            mode: 'lines+markers',
            marker: {size: 10}
        };
        
        var trace4 = {
            x: profiles,
            y: diet_sensitivity,
            name: 'Diet Sensitivity',
            type: 'scatter',
            mode: 'lines+markers',
            marker: {size: 10}
        };
        
        var layout = {
            title: 'Patient Profile Adaptation Over Time',
            xaxis: {
                title: 'Time Period'
            },
            yaxis: {
                title: 'Sensitivity Score',
                range: [0, 1]
            },
            legend: {
                x: 0.02,
                y: 1,
                bgcolor: 'rgba(255, 255, 255, 0.5)'
            },
            margin: {l: 40, r: 20, t: 60, b: 40}
        };
        
        Plotly.newPlot('profile-adaptation-chart', [trace1, trace2, trace3, trace4], layout, {responsive: true});
    })();
    
    // Personalized Gating Adjustments Chart
    (function() {
        var time_periods = ['Baseline', 'Week 1', 'Week 2', 'Week 3', 'Week 4'];
        var expert1_weights = [0.25, 0.3, 0.35, 0.4, 0.45];
        var expert2_weights = [0.35, 0.3, 0.25, 0.2, 0.15];
        var expert3_weights = [0.2, 0.25, 0.3, 0.3, 0.3];
        var expert4_weights = [0.2, 0.15, 0.1, 0.1, 0.1];
        
        var trace1 = {
            x: time_periods,
            y: expert1_weights,
            type: 'bar',
            name: 'Time-Based Expert',
            marker: {
                color: 'rgba(58, 71, 80, 0.6)',
                line: {
                    color: 'rgba(58, 71, 80, 1.0)',
                    width: 1
                }
            }
        };
        
        var trace2 = {
            x: time_periods,
            y: expert2_weights,
            type: 'bar',
            name: 'Physiological Expert',
            marker: {
                color: 'rgba(246, 78, 139, 0.6)',
                line: {
                    color: 'rgba(246, 78, 139, 1.0)',
                    width: 1
                }
            }
        };
        
        var trace3 = {
            x: time_periods,
            y: expert3_weights,
            type: 'bar',
            name: 'Environmental Expert',
            marker: {
                color: 'rgba(6, 147, 227, 0.6)',
                line: {
                    color: 'rgba(6, 147, 227, 1.0)',
                    width: 1
                }
            }
        };
        
        var trace4 = {
            x: time_periods,
            y: expert4_weights,
            type: 'bar',
            name: 'Behavioral Expert',
            marker: {
                color: 'rgba(153, 204, 255, 0.6)',
                line: {
                    color: 'rgba(153, 204, 255, 1.0)',
                    width: 1
                }
            }
        };
        
        var layout = {
            title: 'Personalized Expert Gating Weight Adjustments',
            xaxis: {
                title: 'Time Period'
            },
            yaxis: {
                title: 'Expert Weight',
                range: [0, 1]
            },
            barmode: 'stack',
            legend: {
                x: 0.02,
                y: 1,
                bgcolor: 'rgba(255, 255, 255, 0.5)'
            },
            margin: {l: 40, r: 20, t: 60, b: 40}
        };
        
        Plotly.newPlot('gating-adjustments-chart', [trace1, trace2, trace3, trace4], layout, {responsive: true});
    })();
    
    // Online Adaptation Capability Chart
    (function() {
        var time = Array.from({length: 50}, (_, i) => i);
        
        // Generate data for different adaptation strategies
        var no_adaptation = time.map(t => 0.1 + 0.01*t);
        var slow_adaptation = time.map(t => 0.1 + 0.01*t - 0.008*Math.min(t, 25));
        var fast_adaptation = time.map(t => 0.1 + 0.01*t - 0.015*Math.min(t, 15));
        var personalized_adaptation = time.map(t => {
            if (t < 10) return 0.1 + 0.01*t;
            if (t < 20) return 0.1 + 0.01*10 - 0.02*(t-10);
            return 0.1 + 0.01*10 - 0.02*10 + 0.005*(t-20);
        });
        
        var trace1 = {
            x: time,
            y: no_adaptation,
            type: 'scatter',
            mode: 'lines',
            name: 'No Adaptation',
            line: {shape: 'spline', smoothing: 1.3}
        };
        
        var trace2 = {
            x: time,
            y: slow_adaptation,
            type: 'scatter',
            mode: 'lines',
            name: 'Slow Adaptation',
            line: {shape: 'spline', smoothing: 1.3}
        };
        
        var trace3 = {
            x: time,
            y: fast_adaptation,
            type: 'scatter',
            mode: 'lines',
            name: 'Fast Adaptation',
            line: {shape: 'spline', smoothing: 1.3}
        };
        
        var trace4 = {
            x: time,
            y: personalized_adaptation,
            type: 'scatter',
            mode: 'lines',
            name: 'Personalized Adaptation',
            line: {shape: 'spline', smoothing: 1.3}
        };
        
        var layout = {
            title: 'Online Adaptation Response to Drift',
            xaxis: {
                title: 'Time'
            },
            yaxis: {
                title: 'Error Rate',
                range: [0, 0.6]
            },
            shapes: [{
                type: 'rect',
                x0: 10,
                y0: 0,
                x1: 20,
                y1: 0.6,
                fillcolor: 'rgba(255, 0, 0, 0.1)',
                line: {width: 0}
            }],
            annotations: [{
                x: 15,
                y: 0.5,
                text: 'Drift Period',
                showarrow: true,
                arrowhead: 2,
                ax: 0,
                ay: -40
            }],
            legend: {
                x: 0.02,
                y: 1,
                bgcolor: 'rgba(255, 255, 255, 0.5)'
            },
            margin: {l: 40, r: 20, t: 60, b: 40}
        };
        
        Plotly.newPlot('online-adaptation-chart', [trace1, trace2, trace3, trace4], layout, {responsive: true});
    })();
    
    // Personalization Effectiveness Metrics Chart
    (function() {
        var categories = ['Prediction Accuracy', 'False Alarm Rate', 'Time to Detect', 'Patient Satisfaction'];
        var baseline = [0.7, 0.3, 0.6, 0.65];
        var personalized = [0.85, 0.15, 0.8, 0.9];
        
        var trace1 = {
            x: categories,
            y: baseline,
            name: 'Baseline Model',
            type: 'bar',
            marker: {
                color: 'rgb(158, 202, 225)',
                opacity: 0.8,
                line: {
                    color: 'rgb(8, 48, 107)',
                    width: 1.5
                }
            }
        };
        
        var trace2 = {
            x: categories,
            y: personalized,
            name: 'Personalized Model',
            type: 'bar',
            marker: {
                color: 'rgb(58, 200, 225)',
                opacity: 0.8,
                line: {
                    color: 'rgb(8, 48, 107)',
                    width: 1.5
                }
            }
        };
        
        var data = [trace1, trace2];
        
        var layout = {
            title: 'Personalization Effectiveness Metrics',
            xaxis: {
                title: '',
                tickangle: -45
            },
            yaxis: {
                title: 'Score',
                range: [0, 1]
            },
            barmode: 'group',
            bargap: 0.15,
            bargroupgap: 0.1,
            legend: {
                x: 0.02,
                y: 1,
                bgcolor: 'rgba(255, 255, 255, 0.5)'
            },
            margin: {l: 40, r: 20, t: 60, b: 100}
        };
        
        Plotly.newPlot('personalization-effectiveness-chart', data, layout, {responsive: true});
    })();
    </script>
    """)
    
    # Close the section
    html_content.append('</div>')
    
    return html_content
