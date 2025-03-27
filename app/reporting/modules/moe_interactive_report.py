"""
Interactive MoE Report Generator

This module provides interactive report generation functionality for the MoE framework.
"""

import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template

def format_metric(value):
    """Format a metric value for display."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def generate_interactive_report(test_results, results_dir=None, return_html=False):
    """
    Generate an interactive HTML report from test results.
    
    Args:
        test_results: Dictionary containing test results and metrics
        results_dir: Directory to save the report (if return_html is False)
        return_html: If True, return HTML content instead of writing to file
        
    Returns:
        Path to the generated report or HTML content if return_html=True
    """
    # Create report sections based on test results
    sections = []
    
    # 1. Overall Performance
    if "end_to_end_performance" in test_results:
        metrics = test_results["end_to_end_performance"].get("metrics", {})
        sections.append({
            "title": "Overall Performance",
            "content": f"""
                <div class='metrics-grid'>
                    <div class='metric-card'>
                        <h3>RMSE</h3>
                        <p class='value'>{format_metric(metrics.get('rmse', 0)):.3f}</p>
                    </div>
                    <div class='metric-card'>
                        <h3>MAE</h3>
                        <p class='value'>{format_metric(metrics.get('mae', 0)):.3f}</p>
                    </div>
                    <div class='metric-card'>
                        <h3>R²</h3>
                        <p class='value'>{format_metric(metrics.get('r2', 0)):.3f}</p>
                    </div>
                </div>
            """
        })
    
    # 2. Expert Benchmarks
    if "expert_benchmarks" in test_results:
        sections.append({
            "title": "Expert Model Performance",
            "content": create_expert_comparison_chart(test_results["expert_benchmarks"])
        })
    
    # 3. Gating Network Analysis
    if "gating_evaluation" in test_results:
        gating = test_results["gating_evaluation"]
        try:
            gating_metrics = {
                "optimal_selection_rate": format_metric(gating.get("optimal_selection_rate", 0)),
                "mean_regret": format_metric(gating.get("mean_regret", 0))
            }
            
            gating_content = []
            
            # Add metrics grid if we have valid data
            if any(v > 0 for v in gating_metrics.values()):
                gating_content.append(f"""
                    <div class='metrics-grid'>
                        <div class='metric-card'>
                            <h3>Optimal Selection Rate</h3>
                            <p class='value'>{gating_metrics['optimal_selection_rate']:.1%}</p>
                        </div>
                        <div class='metric-card'>
                            <h3>Mean Regret</h3>
                            <p class='value'>{gating_metrics['mean_regret']:.3f}</p>
                        </div>
                    </div>
                """)
            
            # Add selection frequencies chart if available
            if "selection_frequencies" in gating and gating["selection_frequencies"]:
                selection_chart = create_expert_selection_chart(gating["selection_frequencies"])
                if selection_chart:
                    gating_content.append(selection_chart)
            
            if gating_content:
                sections.append({
                    "title": "Gating Network Analysis",
                    "content": "\n".join(gating_content)
                })
        except Exception as e:
            logger.warning(f"Error processing gating network analysis: {e}")
            sections.append({
                "title": "Gating Network Analysis",
                "content": "<p>Error processing gating network data. Please check the checkpoint format.</p>"
            })
    
    # Generate HTML report
    html_content = create_html_report(sections)
    
    if return_html:
        return html_content
    else:
        # Save report to file
        output_path = os.path.join(results_dir, "moe_interactive_report.html")
        with open(output_path, "w") as f:
            f.write(html_content)
        return output_path

def create_expert_comparison_chart(expert_data):
    """Create a plotly chart comparing expert performance."""
    # Convert expert data to DataFrame format
    rows = []
    metrics_to_plot = ['rmse', 'mae', 'r2']
    
    for expert_name, metrics in expert_data.items():
        try:
            row = {
                'name': expert_name.replace('_expert', '').title(),
                **{metric: format_metric(metrics.get(metric, 0)) for metric in metrics_to_plot}
            }
            rows.append(row)
        except Exception as e:
            logger.warning(f"Error processing expert {expert_name}: {e}")
            continue
    
    if not rows:
        return "<p>No expert data available for visualization.</p>"
    
    df = pd.DataFrame(rows)
    fig = go.Figure()
    
    # Custom colors for each expert
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['rmse'], row['mae'], row['r2']],
            theta=['RMSE', 'MAE', 'R²'],
            fill='toself',
            name=row['name'],
            line_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(1.0, df[metrics_to_plot].max().max())]
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig.to_html(include_plotlyjs=True, full_html=False)

def create_expert_selection_chart(frequencies):
    """Create a bar chart of expert selection frequencies."""
    try:
        # Convert frequencies to DataFrame
        data = []
        for expert, freq in frequencies.items():
            try:
                data.append({
                    "Expert": expert.replace('_expert', '').title(),
                    "Frequency": format_metric(freq)
                })
            except Exception as e:
                logger.warning(f"Error processing frequency for {expert}: {e}")
                continue
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        
        # Create bar chart
        fig = px.bar(
            df,
            x='Expert',
            y='Frequency',
            title='Expert Selection Frequencies',
            text=df['Frequency'].apply(lambda x: f'{x:.1%}')
        )
        
        # Update styling
        fig.update_traces(
            textposition='outside',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(df)]
        )
        
        fig.update_layout(
            xaxis_title='Expert Model',
            yaxis_title='Selection Frequency',
            yaxis_tickformat=',.0%',
            showlegend=False,
            margin=dict(l=60, r=20, t=40, b=40),
            yaxis=dict(
                range=[0, max(1.0, df['Frequency'].max() * 1.1)]
            )
        )
        
        return fig.to_html(include_plotlyjs=False, full_html=False)
    except Exception as e:
        logger.warning(f"Error creating expert selection chart: {e}")
        return None

def create_html_report(sections):
    """Create the complete HTML report."""
    template = Template("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MoE Performance Analysis</title>
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
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .metric-card h3 {
                margin: 0;
                color: #7f8c8d;
                font-size: 0.9em;
            }
            .metric-card .value {
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
                color: #2c3e50;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MoE Framework Performance Analysis</h1>
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
