"""
Publication-ready visualizations for MoE framework.

This module provides functions for creating publication-ready visualizations 
for the MoE framework, including expert contribution visualizations, k-fold
validation visualizations, and ablation study visualizations.

Each function returns a plotly figure object that can be displayed in the dashboard
or exported as a high-resolution image for publication.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots
# Try to import scipy, but provide fallbacks if not available
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    # Create a stats replacement class for basic functionality
    class StatsReplacement:
        def pearsonr(self, x, y):
            # Return dummy correlation (0.5) and p-value (0.1)
            return 0.5, 0.1
            
        def ttest_ind(self, a, b, equal_var=True):
            # Return dummy statistic (0.5) and p-value (0.1)
            return 0.5, 0.1
    
    stats = StatsReplacement()
    print("Warning: scipy not available. Using simplified stats calculations.")
import json
import datetime
import streamlit as st


def export_publication_figure(fig, filename, format="png", width=1200, height=800, scale=2, 
                             include_plotly_logo=False, include_timestamp=True, journal="generic"):
    """
    Export a plotly figure as a publication-ready image.
    
    Args:
        fig: A plotly figure object
        filename: The filename to save the image as
        format: The image format (png, jpg, pdf, svg)
        width: The width of the image in pixels
        height: The height of the image in pixels
        scale: The scale factor for the image (2 = 2x resolution)
        include_plotly_logo: Whether to include the plotly logo
        include_timestamp: Whether to include a timestamp in the filename
        journal: The journal format to use (generic, IEEE, APA)

    Returns:
        The path to the saved image
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Apply journal-specific formatting
    if journal == "IEEE":
        fig.update_layout(
            font=dict(family="Times New Roman", size=12),
            width=width, height=height,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
    elif journal == "APA":
        fig.update_layout(
            font=dict(family="Arial", size=12),
            width=width, height=height,
            margin=dict(l=60, r=50, t=80, b=60),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
    else:  # Generic format
        fig.update_layout(
            width=width, height=height,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
    
    # Add timestamp to filename if requested
    if include_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = filename.rsplit('.', 1)
        if len(filename_parts) == 2:
            filename = f"{filename_parts[0]}_{timestamp}.{filename_parts[1]}"
        else:
            filename = f"{filename}_{timestamp}"
    
    # Ensure filename has the correct extension
    if not filename.endswith(f".{format}"):
        filename = f"{filename}.{format}"
    
    # Export the figure
    fig.write_image(
        filename, 
        width=width, 
        height=height, 
        scale=scale,
        engine="orca" if format == "pdf" else "kaleido"
    )
    
    return filename


def create_expert_contribution_heatmap(data, title="Expert Contribution Heatmap"):
    """
    Create a heatmap visualization of expert contributions.
    
    Args:
        data: A pandas DataFrame with expert contributions, with columns:
              - expert_name: The name of the expert
              - sample_id or time: The sample ID or time point
              - contribution: The contribution value (0-1)
        title: The title of the plot
        
    Returns:
        A plotly figure object
    """
    # Check if we need to pivot the data
    if isinstance(data, pd.DataFrame) and 'expert_name' in data.columns:
        # Determine which column to use as the x-axis
        x_col = 'sample_id' if 'sample_id' in data.columns else 'time' if 'time' in data.columns else None
        
        if x_col is None:
            raise ValueError("Data must contain a 'sample_id' or 'time' column")
        
        # Pivot the data to create a heatmap
        pivot_data = data.pivot(index='expert_name', columns=x_col, values='contribution')
    else:
        # Assume data is already in the correct format
        pivot_data = data
        
    # Create heatmap
    fig = px.imshow(
        pivot_data, 
        color_continuous_scale="YlGnBu",
        labels=dict(x="Sample", y="Expert", color="Contribution"),
        title=title
    )
    
    # Set layout for publication quality
    fig.update_layout(
        font=dict(family="Arial", size=14),
        coloraxis_colorbar=dict(
            title="Weight",
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["0", "0.25", "0.5", "0.75", "1"],
            lenmode="fraction", len=0.75
        ),
        margin=dict(l=60, r=40, t=80, b=50),
        xaxis_nticks=20,
        xaxis=dict(title="Sample/Time Point"),
        yaxis=dict(title="Expert Model")
    )
    
    return fig


def create_expert_weights_timeline(data, title="Expert Weight Evolution Over Time"):
    """
    Create a line chart visualization of expert weights over time.
    
    Args:
        data: A pandas DataFrame with columns:
              - time: The time point (x-axis)
              - expert columns: One column per expert with weight values
        title: The title of the plot
        
    Returns:
        A plotly figure object
    """
    # Check data format
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    # Determine the time column
    time_cols = [col for col in data.columns if any(t in col.lower() for t in ['time', 'timestamp', 'date'])]
    if time_cols:
        time_col = time_cols[0]
    elif 'sample_id' in data.columns:
        time_col = 'sample_id'
    else:
        raise ValueError("Data must contain a time or sample_id column")
    
    # Get expert columns (all columns except time column)
    expert_cols = [col for col in data.columns if col != time_col]
    
    if not expert_cols:
        raise ValueError("No expert columns found in data")
    
    # Create line chart
    fig = go.Figure()
    
    for expert in expert_cols:
        fig.add_trace(go.Scatter(
            x=data[time_col],
            y=data[expert],
            mode='lines',
            name=expert,
            line=dict(width=3)
        ))
    
    # Set layout for publication quality
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Expert Weight",
        font=dict(family="Arial", size=14),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        yaxis=dict(range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    )
    
    # Add a horizontal line at y=0.5 for reference
    fig.add_shape(
        type="line",
        x0=data[time_col].min(),
        y0=0.5,
        x1=data[time_col].max(),
        y1=0.5,
        line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash")
    )
    
    return fig


def create_ablation_study_chart(data, title="Ablation Study Results", metric="RMSE"):
    """
    Create a bar chart visualization of ablation study results.
    
    Args:
        data: A pandas DataFrame with columns:
              - model_configuration: The expert combination description
              - performance_metric: The performance metric value
        title: The title of the plot
        metric: The name of the performance metric
        
    Returns:
        A plotly figure object
    """
    # Create bar chart
    fig = px.bar(
        data,
        x='model_configuration', 
        y='performance_metric',
        color='performance_metric',
        labels={
            'model_configuration': 'Expert Configuration',
            'performance_metric': metric
        },
        title=title,
        color_continuous_scale='Viridis'
    )
    
    # Set layout for publication quality
    fig.update_layout(
        font=dict(family="Arial", size=14),
        coloraxis_showscale=False,
        xaxis_tickangle=-45,
        margin=dict(l=60, r=40, t=80, b=120),
        yaxis_title=metric
    )
    
    # Add p-value annotations if available
    if 'p_value' in data.columns and 'significance' in data.columns:
        for i, row in data.iterrows():
            if row['significance']:
                fig.add_annotation(
                    x=row['model_configuration'],
                    y=row['performance_metric'],
                    text=f"p={row['p_value']:.3f}*",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=12)
                )
    
    return fig


def create_kfold_validation_chart(data, title="K-Fold Cross-Validation Results"):
    """
    Create a box plot visualization of k-fold validation results.
    
    Args:
        data: A pandas DataFrame with columns:
              - model: The model name
              - fold: The fold number
              - metric columns: One or more metric columns
        title: The title of the plot
        
    Returns:
        A plotly figure object
    """
    # Check data format
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    # Determine metric columns (all numeric columns except model and fold)
    metric_cols = [col for col in data.columns 
                  if col not in ['model', 'fold'] 
                  and pd.api.types.is_numeric_dtype(data[col])]
    
    if not metric_cols:
        raise ValueError("No metric columns found in data")
    
    # Create box plot
    fig = go.Figure()
    
    # Add a box plot for each model
    for model_name in data['model'].unique():
        model_data = data[data['model'] == model_name]
        
        for metric in metric_cols:
            fig.add_trace(go.Box(
                y=model_data[metric],
                name=f"{model_name}<br>{metric}",
                boxmean=True,  # Show mean as a dashed line
                marker_color=px.colors.qualitative.Plotly[metric_cols.index(metric) % len(px.colors.qualitative.Plotly)],
                line=dict(width=2)
            ))
    
    # Set layout for publication quality
    fig.update_layout(
        title=title,
        yaxis_title="Metric Value",
        xaxis_title="Model & Metric",
        font=dict(family="Arial", size=14),
        boxmode='group',
        margin=dict(l=60, r=40, t=80, b=120),
        xaxis_tickangle=-45
    )
    
    # Add mean values as text annotations
    for i, model_name in enumerate(data['model'].unique()):
        model_data = data[data['model'] == model_name]
        
        for j, metric in enumerate(metric_cols):
            mean_val = model_data[metric].mean()
            std_val = model_data[metric].std()
            
            fig.add_annotation(
                x=i + j/(len(metric_cols)+1),
                y=mean_val,
                text=f"Mean: {mean_val:.3f}<br>Std: {std_val:.3f}",
                showarrow=False,
                yshift=20,
                font=dict(size=10)
            )
    
    return fig


def create_patient_subgroup_analysis(data, title="Patient Subgroup Performance Analysis"):
    """
    Create visualization of model performance across different patient subgroups.
    
    Args:
        data: A pandas DataFrame with columns:
              - subgroup: The patient subgroup
              - model: The model name
              - metric columns: One or more metric columns
        title: The title of the plot
        
    Returns:
        A plotly figure object
    """
    # Check data format
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    required_cols = ['subgroup', 'model']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Data must contain a '{col}' column")
    
    # Determine metric columns (all numeric columns except required columns)
    metric_cols = [col for col in data.columns 
                  if col not in required_cols 
                  and pd.api.types.is_numeric_dtype(data[col])]
    
    if not metric_cols:
        raise ValueError("No metric columns found in data")
    
    # Use the first metric column by default
    default_metric = metric_cols[0]
    
    # Create a grouped bar chart
    fig = px.bar(
        data,
        x='subgroup',
        y=default_metric,
        color='model',
        barmode='group',
        title=title,
        labels={
            'subgroup': 'Patient Subgroup',
            default_metric: default_metric
        }
    )
    
    # Set layout for publication quality
    fig.update_layout(
        font=dict(family="Arial", size=14),
        xaxis_tickangle=-45,
        margin=dict(l=60, r=40, t=80, b=120),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        )
    )
    
    # Add a hover template with all metrics
    hovertemplate = "<b>%{x}</b><br>%{fullData.name}<br>"
    for metric in metric_cols:
        hovertemplate += f"{metric}: %{{customdata[{metric_cols.index(metric)}]:.3f}}<br>"
    hovertemplate += "<extra></extra>"
    
    # Update traces with custom data and hover template
    for i, trace in enumerate(fig.data):
        model_name = trace.name
        subgroup_model_data = data[(data['model'] == model_name)]
        
        # Create custom data array with all metrics
        custom_data = []
        for _, row in subgroup_model_data.iterrows():
            custom_data.append([row[metric] for metric in metric_cols])
        
        fig.data[i].customdata = custom_data
        fig.data[i].hovertemplate = hovertemplate
    
    return fig


def load_ablation_study_results(base_dir="./output/moe_validation"):
    """
    Load ablation study results from validation files.
    
    Args:
        base_dir: The base directory containing validation results
        
    Returns:
        A pandas DataFrame with ablation study results
    """
    results_data = []
    
    # Find validation result files
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_ablation_study.json') or file.endswith('validation_results.json'):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        results = json.load(f)
                    
                    # Check if this file has ablation study results
                    if 'ablation_study' in results:
                        ablation_data = results['ablation_study']
                        
                        # Process data into a standard format
                        for config, metrics in ablation_data.items():
                            row = {
                                'model_configuration': config,
                                'file': file
                            }
                            
                            # Add all metrics
                            for metric_name, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    row[metric_name] = value
                            
                            results_data.append(row)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
    
    # Create DataFrame
    if results_data:
        df = pd.DataFrame(results_data)
        
        # If there's a primary metric column, rename it to performance_metric
        metric_cols = [col for col in df.columns if col not in ['model_configuration', 'file', 'p_value', 'significance']]
        if metric_cols and 'performance_metric' not in df.columns:
            primary_metric = metric_cols[0]  # Use first metric as primary
            df['performance_metric'] = df[primary_metric]
            
        return df
    else:
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['model_configuration', 'performance_metric', 'file'])


def load_kfold_validation_results(base_dir="./output/moe_validation"):
    """
    Load k-fold validation results from validation files.
    
    Args:
        base_dir: The base directory containing validation results
        
    Returns:
        A pandas DataFrame with k-fold validation results
    """
    results_data = []
    
    # Find validation result files
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_kfold.json') or file.endswith('_cross_validation.json'):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        results = json.load(f)
                    
                    # Check if this file has k-fold results
                    if 'fold_results' in results:
                        fold_data = results['fold_results']
                        
                        # Process each fold
                        for fold_idx, fold_results in enumerate(fold_data):
                            for model_name, metrics in fold_results.items():
                                row = {
                                    'model': model_name,
                                    'fold': fold_idx + 1,
                                    'file': file
                                }
                                
                                # Add all metrics
                                for metric_name, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        row[metric_name] = value
                                
                                results_data.append(row)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
    
    # Create DataFrame
    if results_data:
        return pd.DataFrame(results_data)
    else:
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['model', 'fold', 'file'])


def generate_publication_figures(data_dict, output_dir="./publications"):
    """
    Generate publication-ready figures based on provided data.
    
    Args:
        data_dict: A dictionary with data for different visualization types:
                   - expert_contributions
                   - expert_weights
                   - ablation_study
                   - kfold_validation
                   - patient_subgroups
        output_dir: The directory to save figures to
        
    Returns:
        A list of paths to saved figures
    """
    saved_paths = []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate expert contribution heatmap
    if 'expert_contributions' in data_dict and data_dict['expert_contributions'] is not None:
        fig = create_expert_contribution_heatmap(
            data_dict['expert_contributions'],
            title="Expert Contribution Analysis"
        )
        path = os.path.join(output_dir, "expert_contributions.png")
        saved_path = export_publication_figure(fig, path, format="png", journal="IEEE")
        saved_paths.append(saved_path)
    
    # Generate expert weights timeline
    if 'expert_weights' in data_dict and data_dict['expert_weights'] is not None:
        fig = create_expert_weights_timeline(
            data_dict['expert_weights'],
            title="Expert Weight Evolution Over Time"
        )
        path = os.path.join(output_dir, "expert_weights_timeline.png")
        saved_path = export_publication_figure(fig, path, format="png", journal="IEEE")
        saved_paths.append(saved_path)
    
    # Generate ablation study chart
    if 'ablation_study' in data_dict and data_dict['ablation_study'] is not None:
        fig = create_ablation_study_chart(
            data_dict['ablation_study'],
            title="Expert Ablation Study",
            metric=data_dict.get('ablation_metric', 'RMSE')
        )
        path = os.path.join(output_dir, "ablation_study.png")
        saved_path = export_publication_figure(fig, path, format="png", journal="IEEE")
        saved_paths.append(saved_path)
    
    # Generate k-fold validation chart
    if 'kfold_validation' in data_dict and data_dict['kfold_validation'] is not None:
        fig = create_kfold_validation_chart(
            data_dict['kfold_validation'],
            title="K-Fold Cross-Validation Results"
        )
        path = os.path.join(output_dir, "kfold_validation.png")
        saved_path = export_publication_figure(fig, path, format="png", journal="IEEE")
        saved_paths.append(saved_path)
    
    # Generate patient subgroup analysis
    if 'patient_subgroups' in data_dict and data_dict['patient_subgroups'] is not None:
        fig = create_patient_subgroup_analysis(
            data_dict['patient_subgroups'],
            title="Patient Subgroup Performance Analysis"
        )
        path = os.path.join(output_dir, "patient_subgroup_analysis.png")
        saved_path = export_publication_figure(fig, path, format="png", journal="IEEE")
        saved_paths.append(saved_path)
    
    return saved_paths 