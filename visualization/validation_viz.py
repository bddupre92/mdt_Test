"""
Validation visualization module for MoE framework.

This module provides specialized visualization functions for the MoE framework's 
validation results, including k-fold cross-validation visualizations, 
ablation studies, and model comparison visuals.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime


def load_validation_reports(base_dir="./output/moe_validation", report_pattern=r".*validation.*\.json"):
    """
    Load validation report files from the specified directory.
    
    Args:
        base_dir: The base directory to search for validation reports
        report_pattern: A regex pattern for matching validation report files
        
    Returns:
        A list of dictionaries, each containing a validation report
    """
    reports = []
    
    try:
        # Create regex pattern
        pattern = re.compile(report_pattern)
        
        # Find validation report files
        for root, _, files in os.walk(base_dir):
            for file in files:
                if pattern.match(file) and file.endswith('.json'):
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            report_data = json.load(f)
                        
                        # Add file metadata
                        report_data['_file'] = os.path.join(root, file)
                        report_data['_filename'] = file
                        report_data['_timestamp'] = os.path.getmtime(os.path.join(root, file))
                        
                        reports.append(report_data)
                    except Exception as e:
                        st.error(f"Error loading validation report {file}: {str(e)}")
    
    except Exception as e:
        st.error(f"Error searching for validation reports: {str(e)}")
    
    # Sort by timestamp (newest first)
    reports.sort(key=lambda x: x.get('_timestamp', 0), reverse=True)
    
    return reports


def extract_kfold_results(validation_reports):
    """
    Extract k-fold cross-validation results from validation reports.
    
    Args:
        validation_reports: A list of validation report dictionaries
        
    Returns:
        A pandas DataFrame with k-fold validation results
    """
    kfold_results = []
    
    for report in validation_reports:
        if 'cross_validation' in report or 'kfold_validation' in report or 'fold_results' in report:
            # Determine which key contains the k-fold results
            if 'cross_validation' in report:
                fold_data = report['cross_validation']
            elif 'kfold_validation' in report:
                fold_data = report['kfold_validation']
            elif 'fold_results' in report:
                fold_data = report['fold_results']
            else:
                continue
            
            # Extract report metadata
            report_file = report.get('_filename', 'unknown')
            timestamp = report.get('_timestamp', 0)
            
            # Process fold results
            if isinstance(fold_data, list):
                # Format: List of dictionaries, each containing results for one fold
                for fold_idx, fold_results in enumerate(fold_data):
                    if isinstance(fold_results, dict):
                        for model, metrics in fold_results.items():
                            if isinstance(metrics, dict):
                                result = {
                                    'report': report_file,
                                    'timestamp': timestamp,
                                    'fold': fold_idx + 1,
                                    'model': model
                                }
                                
                                # Add metrics
                                for metric, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        result[metric] = value
                                
                                kfold_results.append(result)
            
            elif isinstance(fold_data, dict):
                # Format: Dictionary with keys for each model and values containing fold results
                for model, model_data in fold_data.items():
                    if isinstance(model_data, dict):
                        # Check if this is a fold dictionary
                        if any(key.startswith('fold') for key in model_data.keys()):
                            for fold_key, metrics in model_data.items():
                                if fold_key.startswith('fold') and isinstance(metrics, dict):
                                    # Extract fold number
                                    fold_num = int(fold_key.replace('fold', ''))
                                    
                                    result = {
                                        'report': report_file,
                                        'timestamp': timestamp,
                                        'fold': fold_num,
                                        'model': model
                                    }
                                    
                                    # Add metrics
                                    for metric, value in metrics.items():
                                        if isinstance(value, (int, float)):
                                            result[metric] = value
                                    
                                    kfold_results.append(result)
                        
                        # Check if this is a metrics dictionary with a 'folds' key
                        elif 'folds' in model_data and isinstance(model_data['folds'], list):
                            for fold_idx, fold_metrics in enumerate(model_data['folds']):
                                if isinstance(fold_metrics, dict):
                                    result = {
                                        'report': report_file,
                                        'timestamp': timestamp,
                                        'fold': fold_idx + 1,
                                        'model': model
                                    }
                                    
                                    # Add metrics
                                    for metric, value in fold_metrics.items():
                                        if isinstance(value, (int, float)):
                                            result[metric] = value
                                    
                                    kfold_results.append(result)
    
    # Create DataFrame
    if kfold_results:
        return pd.DataFrame(kfold_results)
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['report', 'timestamp', 'fold', 'model'])


def extract_ablation_results(validation_reports):
    """
    Extract ablation study results from validation reports.
    
    Args:
        validation_reports: A list of validation report dictionaries
        
    Returns:
        A pandas DataFrame with ablation study results
    """
    ablation_results = []
    
    for report in validation_reports:
        if 'ablation_study' in report or 'expert_ablation' in report:
            # Determine which key contains the ablation results
            if 'ablation_study' in report:
                ablation_data = report['ablation_study']
            else:
                ablation_data = report['expert_ablation']
            
            # Extract report metadata
            report_file = report.get('_filename', 'unknown')
            timestamp = report.get('_timestamp', 0)
            
            # Process ablation results
            if isinstance(ablation_data, dict):
                for config, metrics in ablation_data.items():
                    if isinstance(metrics, dict):
                        result = {
                            'report': report_file,
                            'timestamp': timestamp,
                            'configuration': config
                        }
                        
                        # Add metrics
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                result[metric] = value
                            elif isinstance(value, dict) and 'value' in value:
                                # Some reports might store metrics as {value: X, p_value: Y}
                                if isinstance(value['value'], (int, float)):
                                    result[metric] = value['value']
                                    
                                # Store p-value if available
                                if 'p_value' in value and isinstance(value['p_value'], (int, float)):
                                    result[f"{metric}_p_value"] = value['p_value']
                        
                        ablation_results.append(result)
    
    # Create DataFrame
    if ablation_results:
        df = pd.DataFrame(ablation_results)
        
        # Check for significance flags
        if any("_p_value" in col for col in df.columns):
            # For each metric with a p-value, create a significance flag
            for col in [c for c in df.columns if c.endswith("_p_value")]:
                base_metric = col.replace("_p_value", "")
                if base_metric in df.columns:
                    df[f"{base_metric}_significant"] = df[col] < 0.05
        
        return df
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['report', 'timestamp', 'configuration'])


def create_kfold_boxplot(kfold_df, metric='rmse', title=None):
    """
    Create a box plot of k-fold validation results.
    
    Args:
        kfold_df: A pandas DataFrame with k-fold validation results
        metric: The metric to visualize
        title: The title of the plot (optional)
        
    Returns:
        A plotly figure object
    """
    if kfold_df is None or kfold_df.empty:
        return None
    
    # Find the closest matching metric
    available_metrics = [col for col in kfold_df.columns if col.lower() == metric.lower()]
    if not available_metrics:
        # Look for partial matches
        available_metrics = [col for col in kfold_df.columns 
                            if not col.startswith('_') 
                            and col not in ['report', 'timestamp', 'fold', 'model']
                            and metric.lower() in col.lower()]
    
    if not available_metrics:
        st.warning(f"Metric '{metric}' not found in k-fold validation results.")
        return None
    
    # Use the first matching metric
    metric_col = available_metrics[0]
    
    # Automatically determine title if not provided
    if title is None:
        title = f"K-Fold Cross-Validation: {metric_col.upper()}"
    
    # Create box plot
    fig = px.box(
        kfold_df,
        x='model',
        y=metric_col,
        color='model',
        notched=True,  # Show confidence interval on median
        points='all',  # Show all points
        title=title,
        labels={
            'model': 'Model',
            metric_col: metric_col.upper()
        }
    )
    
    # Set layout
    fig.update_layout(
        width=800,
        height=500,
        xaxis_title="Model",
        yaxis_title=metric_col.upper(),
        showlegend=False
    )
    
    # Add mean and std as annotations
    for model in kfold_df['model'].unique():
        model_data = kfold_df[kfold_df['model'] == model]
        mean_val = model_data[metric_col].mean()
        std_val = model_data[metric_col].std()
        
        fig.add_annotation(
            x=model,
            y=mean_val,
            text=f"Mean: {mean_val:.3f}<br>Std: {std_val:.3f}",
            showarrow=False,
            yshift=20,
            font=dict(size=10)
        )
    
    return fig


def create_ablation_barchart(ablation_df, metric='rmse', title=None, sort_by_performance=True):
    """
    Create a bar chart of ablation study results.
    
    Args:
        ablation_df: A pandas DataFrame with ablation study results
        metric: The metric to visualize
        title: The title of the plot (optional)
        sort_by_performance: Whether to sort configurations by performance
        
    Returns:
        A plotly figure object
    """
    if ablation_df is None or ablation_df.empty:
        return None
    
    # Find the closest matching metric
    available_metrics = [col for col in ablation_df.columns if col.lower() == metric.lower()]
    if not available_metrics:
        # Look for partial matches
        available_metrics = [col for col in ablation_df.columns 
                            if not col.startswith('_') 
                            and col not in ['report', 'timestamp', 'configuration']
                            and not col.endswith('_p_value')
                            and not col.endswith('_significant')
                            and metric.lower() in col.lower()]
    
    if not available_metrics:
        st.warning(f"Metric '{metric}' not found in ablation study results.")
        return None
    
    # Use the first matching metric
    metric_col = available_metrics[0]
    
    # Check for p-values and significance flags
    p_value_col = f"{metric_col}_p_value" if f"{metric_col}_p_value" in ablation_df.columns else None
    significance_col = f"{metric_col}_significant" if f"{metric_col}_significant" in ablation_df.columns else None
    
    # Create copy of DataFrame to avoid modifying the original
    df = ablation_df.copy()
    
    # Sort by performance if requested
    if sort_by_performance:
        df = df.sort_values(metric_col)
    
    # Automatically determine title if not provided
    if title is None:
        title = f"Ablation Study: {metric_col.upper()}"
    
    # Create bar chart
    fig = px.bar(
        df,
        x='configuration',
        y=metric_col,
        color=metric_col,
        title=title,
        labels={
            'configuration': 'Expert Configuration',
            metric_col: metric_col.upper()
        },
        color_continuous_scale='RdBu_r'  # Use reverse RdBu so blue is better (assuming lower is better)
    )
    
    # Set layout
    fig.update_layout(
        width=800,
        height=500,
        xaxis_title="Expert Configuration",
        yaxis_title=metric_col.upper(),
        coloraxis_showscale=False,
        xaxis_tickangle=-45
    )
    
    # Add p-value annotations if available
    if p_value_col is not None and significance_col is not None:
        for i, row in df.iterrows():
            if row[significance_col]:
                # Add asterisk for significant results
                fig.add_annotation(
                    x=row['configuration'],
                    y=row[metric_col],
                    text=f"p={row[p_value_col]:.3f}*",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=10)
                )
            elif p_value_col in row:
                # Add p-value without asterisk for non-significant results
                fig.add_annotation(
                    x=row['configuration'],
                    y=row[metric_col],
                    text=f"p={row[p_value_col]:.3f}",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=10)
                )
    
    return fig


def create_model_comparison_radar(validation_df, models=None, metrics=None, title=None):
    """
    Create a radar chart comparing multiple models across various metrics.
    
    Args:
        validation_df: A pandas DataFrame with validation results
        models: List of models to include (or None for all)
        metrics: List of metrics to include (or None for all suitable)
        title: The title of the plot (optional)
        
    Returns:
        A plotly figure object
    """
    if validation_df is None or validation_df.empty:
        return None
    
    # Extract models if not provided
    if models is None:
        models = validation_df['model'].unique().tolist() if 'model' in validation_df.columns else []
        
        # If no models found, try configurations
        if not models and 'configuration' in validation_df.columns:
            models = validation_df['configuration'].unique().tolist()
        
        # Limit to 5 models maximum for readability
        if len(models) > 5:
            models = models[:5]
    
    # Determine model identifier column
    model_col = 'model' if 'model' in validation_df.columns else 'configuration'
    
    # Extract metrics if not provided
    if metrics is None:
        # Get numeric columns excluding non-metric columns
        non_metric_cols = ['report', 'timestamp', 'fold', model_col, 'iteration', 'epoch']
        metrics = [col for col in validation_df.columns 
                 if pd.api.types.is_numeric_dtype(validation_df[col]) 
                 and col not in non_metric_cols
                 and not col.startswith('_')
                 and not col.endswith('_p_value')
                 and not col.endswith('_significant')]
        
        # Limit to 6 metrics maximum for readability
        if len(metrics) > 6:
            metrics = metrics[:6]
    
    # Automatically determine title if not provided
    if title is None:
        title = "Model Comparison Across Metrics"
    
    # Create an empty figure
    fig = go.Figure()
    
    # Add traces for each model
    for model in models:
        # Extract data for this model
        model_data = validation_df[validation_df[model_col] == model]
        
        # Skip if no data
        if model_data.empty:
            continue
        
        # Calculate mean for each metric
        values = []
        for metric in metrics:
            if metric in model_data.columns:
                values.append(model_data[metric].mean())
            else:
                values.append(None)
        
        # Skip if no valid values
        if not any(v is not None for v in values):
            continue
        
        # Add a trace for this model
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=model
        ))
    
    # Set layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Normalized range
            )
        ),
        title=title,
        width=700,
        height=600
    )
    
    return fig


def create_metric_correlation_heatmap(validation_df, metrics=None, title=None):
    """
    Create a heatmap showing correlations between different metrics.
    
    Args:
        validation_df: A pandas DataFrame with validation results
        metrics: List of metrics to include (or None for all suitable)
        title: The title of the plot (optional)
        
    Returns:
        A plotly figure object
    """
    if validation_df is None or validation_df.empty:
        return None
    
    # Extract metrics if not provided
    if metrics is None:
        # Get numeric columns excluding non-metric columns
        non_metric_cols = ['report', 'timestamp', 'fold', 'model', 'configuration', 'iteration', 'epoch']
        metrics = [col for col in validation_df.columns 
                 if pd.api.types.is_numeric_dtype(validation_df[col]) 
                 and col not in non_metric_cols
                 and not col.startswith('_')
                 and not col.endswith('_p_value')
                 and not col.endswith('_significant')]
        
        # Limit to 10 metrics maximum for readability
        if len(metrics) > 10:
            metrics = metrics[:10]
    
    # Skip if fewer than 2 metrics
    if len(metrics) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = validation_df[metrics].corr()
    
    # Automatically determine title if not provided
    if title is None:
        title = "Metric Correlation Heatmap"
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        x=metrics,
        y=metrics,
        color_continuous_scale='RdBu_r',
        title=title,
        labels=dict(color="Correlation")
    )
    
    # Set layout
    fig.update_layout(
        width=700,
        height=700,
        xaxis=dict(tickangle=-45),
        coloraxis_colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1", "-0.5", "0", "0.5", "1"]
        )
    )
    
    # Add correlation values as text
    for i, row in enumerate(corr_matrix.index):
        for j, col in enumerate(corr_matrix.columns):
            value = corr_matrix.iloc[i, j]
            fig.add_annotation(
                x=col,
                y=row,
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(value) > 0.5 else "black")
            )
    
    return fig


def create_validation_summary_dashboard(validation_reports):
    """
    Create a dashboard with validation result visualizations.
    
    Args:
        validation_reports: A list of validation report dictionaries
        
    Returns:
        A list of tuples (title, figure) with visualizations
    """
    figures = []
    
    if not validation_reports:
        st.warning("No validation reports found.")
        return figures
    
    # Extract k-fold results
    kfold_df = extract_kfold_results(validation_reports)
    
    if not kfold_df.empty:
        # Identify available metrics
        metric_cols = [col for col in kfold_df.columns 
                      if not col.startswith('_') 
                      and col not in ['report', 'timestamp', 'fold', 'model']]
        
        if metric_cols:
            # Create k-fold box plot for each metric
            for metric in metric_cols[:3]:  # Limit to 3 metrics
                fig = create_kfold_boxplot(kfold_df, metric=metric)
                if fig:
                    figures.append((f"K-Fold Validation: {metric.upper()}", fig))
            
            # Create model comparison radar chart
            fig = create_model_comparison_radar(kfold_df)
            if fig:
                figures.append(("Model Comparison", fig))
            
            # Create metric correlation heatmap
            fig = create_metric_correlation_heatmap(kfold_df)
            if fig:
                figures.append(("Metric Correlation", fig))
    
    # Extract ablation results
    ablation_df = extract_ablation_results(validation_reports)
    
    if not ablation_df.empty:
        # Identify available metrics
        metric_cols = [col for col in ablation_df.columns 
                      if not col.startswith('_') 
                      and col not in ['report', 'timestamp', 'configuration']
                      and not col.endswith('_p_value')
                      and not col.endswith('_significant')]
        
        if metric_cols:
            # Create ablation bar chart for each metric
            for metric in metric_cols[:3]:  # Limit to 3 metrics
                fig = create_ablation_barchart(ablation_df, metric=metric)
                if fig:
                    figures.append((f"Ablation Study: {metric.upper()}", fig))
    
    return figures


def generate_validation_report_pdf(validation_reports, output_file=None):
    """
    Generate a PDF report summarizing validation results.
    
    Args:
        validation_reports: A list of validation report dictionaries
        output_file: The output PDF file path (or None to generate automatically)
        
    Returns:
        The path to the generated PDF file
    """
    import os
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from io import BytesIO
    
    # Generate output file name if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"validation_report_{timestamp}.pdf"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading1_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create document content
    content = []
    
    # Add title
    content.append(Paragraph("MoE Validation Report", title_style))
    content.append(Spacer(1, 12))
    
    # Add date
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    content.append(Spacer(1, 24))
    
    # Extract k-fold results
    kfold_df = extract_kfold_results(validation_reports)
    
    if not kfold_df.empty:
        # Add k-fold validation section
        content.append(Paragraph("K-Fold Cross-Validation Results", heading1_style))
        content.append(Spacer(1, 12))
        
        # Add summary table
        if not kfold_df.empty:
            # Calculate mean and std for each model and metric
            summary_data = []
            
            # Get models and metrics
            models = kfold_df['model'].unique()
            metric_cols = [col for col in kfold_df.columns 
                          if not col.startswith('_') 
                          and col not in ['report', 'timestamp', 'fold', 'model']]
            
            # Create header row
            header = ["Model"]
            for metric in metric_cols:
                header.extend([f"{metric.upper()} Mean", f"{metric.upper()} Std"])
            
            summary_data.append(header)
            
            # Add data rows
            for model in models:
                model_data = kfold_df[kfold_df['model'] == model]
                row = [model]
                
                for metric in metric_cols:
                    if metric in model_data.columns:
                        mean_val = model_data[metric].mean()
                        std_val = model_data[metric].std()
                        row.extend([f"{mean_val:.4f}", f"{std_val:.4f}"])
                    else:
                        row.extend(["N/A", "N/A"])
                
                summary_data.append(row)
            
            # Create table
            table = Table(summary_data)
            
            # Add table style
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ])
            
            table.setStyle(table_style)
            content.append(table)
            content.append(Spacer(1, 24))
        
        # Add k-fold visualizations
        for metric in metric_cols[:3]:  # Limit to 3 metrics
            fig = create_kfold_boxplot(kfold_df, metric=metric)
            if fig:
                # Save figure to memory
                img_bytes = BytesIO()
                fig.write_image(img_bytes, format='png', width=600, height=400)
                img_bytes.seek(0)
                
                # Add figure to content
                content.append(Paragraph(f"K-Fold Cross-Validation: {metric.upper()}", heading2_style))
                content.append(Spacer(1, 6))
                content.append(Image(img_bytes, width=500, height=300))
                content.append(Spacer(1, 24))
    
    # Extract ablation results
    ablation_df = extract_ablation_results(validation_reports)
    
    if not ablation_df.empty:
        # Add ablation study section
        content.append(Paragraph("Ablation Study Results", heading1_style))
        content.append(Spacer(1, 12))
        
        # Add summary table
        if not ablation_df.empty:
            # Create table data
            summary_data = []
            
            # Get metrics
            metric_cols = [col for col in ablation_df.columns 
                          if not col.startswith('_') 
                          and col not in ['report', 'timestamp', 'configuration']
                          and not col.endswith('_p_value')
                          and not col.endswith('_significant')]
            
            # Create header row
            header = ["Configuration"]
            for metric in metric_cols:
                if f"{metric}_p_value" in ablation_df.columns:
                    header.extend([f"{metric.upper()}", f"{metric.upper()} p-value"])
                else:
                    header.append(f"{metric.upper()}")
            
            summary_data.append(header)
            
            # Add data rows
            for _, row in ablation_df.iterrows():
                table_row = [row['configuration']]
                
                for metric in metric_cols:
                    if metric in row:
                        table_row.append(f"{row[metric]:.4f}")
                        
                        # Add p-value if available
                        if f"{metric}_p_value" in row:
                            p_val = row[f"{metric}_p_value"]
                            is_significant = row.get(f"{metric}_significant", p_val < 0.05)
                            
                            if is_significant:
                                table_row.append(f"{p_val:.4f}*")
                            else:
                                table_row.append(f"{p_val:.4f}")
                    else:
                        table_row.append("N/A")
                        if f"{metric}_p_value" in ablation_df.columns:
                            table_row.append("N/A")
                
                summary_data.append(table_row)
            
            # Create table
            table = Table(summary_data)
            
            # Add table style
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ])
            
            table.setStyle(table_style)
            content.append(table)
            content.append(Spacer(1, 24))
        
        # Add ablation visualizations
        for metric in metric_cols[:3]:  # Limit to 3 metrics
            fig = create_ablation_barchart(ablation_df, metric=metric)
            if fig:
                # Save figure to memory
                img_bytes = BytesIO()
                fig.write_image(img_bytes, format='png', width=600, height=400)
                img_bytes.seek(0)
                
                # Add figure to content
                content.append(Paragraph(f"Ablation Study: {metric.upper()}", heading2_style))
                content.append(Spacer(1, 6))
                content.append(Image(img_bytes, width=500, height=300))
                content.append(Spacer(1, 24))
    
    # Build PDF
    doc.build(content)
    
    return output_file 