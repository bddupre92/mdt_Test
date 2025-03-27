"""
Benchmark Performance Report Generator

This module provides benchmark report generation functionality for the MoE framework.
"""

import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template

def generate_benchmark_report(test_results, results_dir=None, return_html=False):
    """
    Generate a benchmark comparison report.
    
    Args:
        test_results: Dictionary containing test results and metrics
        results_dir: Directory to save the report (if return_html is False)
        return_html: If True, return HTML content instead of writing to file
        
    Returns:
        Path to the generated report or HTML content if return_html=True
    """
    sections = []
    
    # 1. Performance vs Baselines
    if "baseline_comparisons" in test_results:
        sections.append({
            "title": "Performance vs Baseline Models",
            "content": create_baseline_comparison_chart(test_results["baseline_comparisons"])
        })
    
    # 2. Statistical Analysis
    if "statistical_tests" in test_results:
        sections.append({
            "title": "Statistical Analysis",
            "content": create_statistical_analysis_table(test_results["statistical_tests"])
        })
    
    # 3. Expert Model Benchmarks
    if "expert_benchmarks" in test_results:
        sections.append({
            "title": "Expert Model Benchmarks",
            "content": create_expert_benchmark_table(test_results["expert_benchmarks"])
        })
    
    # Generate HTML report
    html_content = create_html_report(sections)
    
    if return_html:
        return html_content
    else:
        output_path = os.path.join(results_dir, "benchmark_performance_report.html")
        with open(output_path, "w") as f:
            f.write(html_content)
        return output_path

def create_baseline_comparison_chart(baseline_data):
    """Create a bar chart comparing MoE with baseline models."""
    df = pd.DataFrame(baseline_data)
    
    fig = px.bar(
        df,
        x='model',
        y=['rmse', 'mae', 'r2'],
        title='Performance vs Baselines',
        barmode='group'
    )
    
    return fig.to_html(include_plotlyjs=True, full_html=False)

def create_statistical_analysis_table(stats_data):
    """Create an HTML table of statistical test results."""
    df = pd.DataFrame(stats_data)
    
    return f"""
        <div class="table-container">
            {df.to_html(classes='styled-table', index=False)}
        </div>
    """

def create_expert_benchmark_table(expert_data):
    """Create an HTML table of expert benchmark results."""
    rows = []
    for expert, metrics in expert_data.items():
        row = {
            "Expert": expert,
            "RMSE": f"{float(metrics['rmse']):.3f}",
            "MAE": f"{float(metrics['mae']):.3f}",
            "R²": f"{float(metrics['r2']):.3f}",
            "Training Time": f"{float(metrics['training_time']):.2f}s"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return f"""
        <div class="table-container">
            {df.to_html(classes='styled-table', index=False)}
        </div>
    """

def create_html_report(sections):
    """Create the complete HTML report."""
    template = Template("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MoE Benchmark Performance Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1, h2 { color: #2c3e50; }
            .section { margin-bottom: 30px; }
            .table-container {
                overflow-x: auto;
                margin: 20px 0;
            }
            .styled-table {
                width: 100%;
                border-collapse: collapse;
                margin: 25px 0;
                font-size: 0.9em;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                border-radius: 5px;
            }
            .styled-table thead tr {
                background-color: #2c3e50;
                color: white;
                text-align: left;
            }
            .styled-table th,
            .styled-table td {
                padding: 12px 15px;
            }
            .styled-table tbody tr {
                border-bottom: 1px solid #dddddd;
            }
            .styled-table tbody tr:nth-of-type(even) {
                background-color: #f3f3f3;
            }
            .styled-table tbody tr:last-of-type {
                border-bottom: 2px solid #2c3e50;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MoE Benchmark Performance Analysis</h1>
            {% for section in sections %}
            <div class="section">
                <h2>{{ section.title }}</h2>
                {{ section.content }}
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """)
    
    return template.render(sections=sections)
