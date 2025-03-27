"""
MOE Publication Report Module

Generates publication-ready visualizations and analysis for the MOE framework.
"""

import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from jinja2 import Template

logger = logging.getLogger(__name__)

def preprocess_data(data_df, target_col):
    """Preprocess data for MOE analysis."""
    # Separate features and target
    X = data_df.drop(columns=[target_col])
    y = data_df[target_col]
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Handle missing values in numeric columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    
    # Handle categorical columns
    X_processed = X[numeric_cols].copy()
    
    # One-hot encode categorical columns
    for col in categorical_cols:
        # Skip patient ID and timestamp-like columns
        if col.lower() in ['patient_id', 'id', 'timestamp', 'date']:
            continue
            
        # Get dummies and add to processed features
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
        X_processed = pd.concat([X_processed, dummies], axis=1)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_processed),
        columns=X_processed.columns,
        index=X_processed.index
    )
    
    return X_scaled, y, scaler

def generate_synthetic_data(real_data, n_samples=1000):
    """Generate synthetic data based on real data distribution."""
    synthetic_data = pd.DataFrame()
    
    # Process each column based on its type
    for column in real_data.columns:
        if column.lower() in ['patient_id', 'id']:
            # Generate sequential IDs
            synthetic_data[column] = [f'S{i:04d}' for i in range(n_samples)]
            
        elif column.lower() in ['timestamp', 'date']:
            # Generate sequential dates
            base_date = pd.Timestamp('2025-01-01')
            synthetic_data[column] = [
                base_date + pd.Timedelta(days=i) for i in range(n_samples)
            ]
            
        elif real_data[column].dtype in ['int64', 'float64']:
            # Generate from normal distribution with bounds
            mean = real_data[column].mean()
            std = real_data[column].std()
            min_val = real_data[column].min()
            max_val = real_data[column].max()
            
            values = np.random.normal(mean, std, n_samples)
            values = np.clip(values, min_val, max_val)
            
            if real_data[column].dtype == 'int64':
                values = values.round().astype('int64')
                
            synthetic_data[column] = values
            
        else:
            # Sample from categorical values maintaining proportions
            value_counts = real_data[column].value_counts(normalize=True)
            synthetic_data[column] = np.random.choice(
                value_counts.index,
                size=n_samples,
                p=value_counts.values
            )
    
    return synthetic_data

def create_expert_performance_plot(expert_data):
    """Create publication-ready expert performance visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'polar'}, {'type': 'domain'}],
            [{'type': 'xy'}, {'type': 'xy'}]
        ],
        subplot_titles=(
            'Expert Model Performance',
            'Expert Selection Distribution',
            'Temporal Performance',
            'Feature Importance'
        )
    )
    
    # 1. Expert Model Performance (Radar Chart)
    experts = []
    metrics = ['RMSE', 'MAE', 'R²']
    for expert_name, metrics_dict in expert_data['expert_benchmarks'].items():
        expert = {
            'name': expert_name.replace('_expert', '').title(),
            'metrics': [
                float(metrics_dict['rmse']),
                float(metrics_dict['mae']),
                float(metrics_dict['r2'])
            ]
        }
        experts.append(expert)
    
    for expert in experts:
        fig.add_trace(
            go.Scatterpolar(
                r=expert['metrics'],
                theta=metrics,
                name=expert['name'],
                fill='toself'
            ),
            row=1, col=1
        )
    
    # 2. Expert Selection Distribution (Pie Chart)
    selections = expert_data['gating_evaluation']['selection_frequencies']
    fig.add_trace(
        go.Pie(
            labels=[k.replace('_expert', '').title() for k in selections.keys()],
            values=[float(v) for v in selections.values()],
            hole=0.3
        ),
        row=1, col=2
    )
    
    # 3. Temporal Performance (Line Chart)
    if 'temporal_metrics' in expert_data:
        temporal = expert_data['temporal_metrics']
        fig.add_trace(
            go.Scatter(
                x=temporal['timestamps'],
                y=temporal['performance'],
                mode='lines+markers',
                name='Performance'
            ),
            row=2, col=1
        )
    
    # 4. Feature Importance (Bar Chart)
    if 'feature_importance' in expert_data:
        importance = expert_data['feature_importance']
        fig.add_trace(
            go.Bar(
                x=list(importance.keys()),
                y=list(importance.values()),
                name='Importance'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white',
        title_text='MOE Framework Analysis',
        title_x=0.5
    )
    
    return fig.to_html(include_plotlyjs=True, full_html=False)

def create_synthetic_comparison_plot(real_data, synthetic_data, target_col):
    """Create visualization comparing real and synthetic data distributions."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Target Distribution',
            'Feature Correlations',
            'PCA Visualization',
            'Feature Distributions'
        )
    )
    
    # 1. Target Distribution
    fig.add_trace(
        go.Histogram(
            x=real_data[target_col],
            name='Real',
            opacity=0.7
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(
            x=synthetic_data[target_col],
            name='Synthetic',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # 2. Feature Correlations
    numeric_cols = real_data.select_dtypes(include=['int64', 'float64']).columns
    real_corr = real_data[numeric_cols].corr()[target_col].sort_values()
    synthetic_corr = synthetic_data[numeric_cols].corr()[target_col].sort_values()
    
    fig.add_trace(
        go.Scatter(
            x=real_corr.index,
            y=real_corr.values,
            mode='markers',
            name='Real Correlations'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=synthetic_corr.index,
            y=synthetic_corr.values,
            mode='markers',
            name='Synthetic Correlations'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white',
        title_text='Real vs Synthetic Data Analysis',
        title_x=0.5
    )
    
    return fig.to_html(include_plotlyjs=True, full_html=False)

def generate_publication_report(test_results, real_data=None, synthetic_data=None, return_html=False):
    """Generate a publication-ready HTML report."""
    sections = []
    
    # 1. MOE Framework Analysis
    if test_results:
        sections.append({
            "title": "MOE Framework Analysis",
            "content": create_expert_performance_plot(test_results)
        })
    
    # 2. Data Analysis
    if real_data is not None and synthetic_data is not None:
        sections.append({
            "title": "Data Analysis",
            "content": create_synthetic_comparison_plot(
                real_data,
                synthetic_data,
                test_results.get('target_column', 'target')
            )
        })
    
    # Generate HTML report
    template = Template("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MOE Framework Analysis Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .section {
                background-color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            h1, h2 {
                color: #333;
            }
            .plot-container {
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <h1>MOE Framework Analysis Report</h1>
        {% for section in sections %}
        <div class="section">
            <h2>{{ section.title }}</h2>
            <div class="plot-container">
                {{ section.content }}
            </div>
        </div>
        {% endfor %}
    </body>
    </html>
    """)
    
    html_content = template.render(sections=sections)
    
    if return_html:
        return html_content
    
    # Save the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        test_results.get('output_dir', 'results'),
        f'moe_publication_report_{timestamp}.html'
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Report saved to: {output_path}")
    return output_path
