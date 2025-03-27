"""
Overall Metrics Component for End-to-End Performance Analysis

This module provides visualization and analysis of the overall performance metrics
for the MoE system.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import logging

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

logger = logging.getLogger(__name__)

def render_overall_metrics(end_to_end_metrics: Dict[str, Any]):
    """
    Render the overall metrics component for end-to-end performance analysis.
    
    Args:
        end_to_end_metrics: Dictionary containing end-to-end performance metrics
    """
    st.subheader("Overall Performance Metrics")
    
    # Extract key metrics
    metrics = extract_key_metrics(end_to_end_metrics)
    
    if not metrics:
        st.info("No overall performance metrics available")
        return
    
    # Display key metrics in a clean layout
    display_key_metrics(metrics)
    
    # Display detailed metrics if available
    detailed_metrics = safe_get_metric(end_to_end_metrics, "detailed_metrics")
    if detailed_metrics is not None and isinstance(detailed_metrics, dict):
        with st.expander("Detailed Metrics", expanded=False):
            display_detailed_metrics(detailed_metrics)
    
    # Display error distribution if available
    error_distribution = safe_get_metric(end_to_end_metrics, "error_distribution")
    if error_distribution is not None and isinstance(error_distribution, dict):
        with st.expander("Error Distribution", expanded=False):
            display_error_distribution(error_distribution)
    
    # Display segmented performance if available
    segmented_performance = safe_get_metric(end_to_end_metrics, "segmented_performance")
    if segmented_performance is not None and isinstance(segmented_performance, dict):
        with st.expander("Performance by Segment", expanded=False):
            display_segmented_performance(segmented_performance)

from app.ui.components.performance_views.helpers import safe_get_metric, safe_has_metric

def extract_key_metrics(end_to_end_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the key metrics from the end-to-end metrics dictionary or object.
    
    Args:
        end_to_end_metrics: Dictionary or object containing end-to-end performance metrics
        
    Returns:
        Dictionary containing key metrics
    """
    metrics = {}
    
    # Extract error metrics
    for key in ["rmse", "mae", "mse", "r2", "adjusted_r2", "mape"]:
        if safe_has_metric(end_to_end_metrics, key):
            metrics[key] = safe_get_metric(end_to_end_metrics, key)
    
    # Extract timing metrics
    for key in ["training_time", "inference_time", "total_processing_time"]:
        if safe_has_metric(end_to_end_metrics, key):
            metrics[key] = safe_get_metric(end_to_end_metrics, key)
    
    # Extract sample counts
    for key in ["sample_count", "training_samples", "test_samples"]:
        if safe_has_metric(end_to_end_metrics, key):
            metrics[key] = safe_get_metric(end_to_end_metrics, key)
    
    return metrics

def display_key_metrics(metrics: Dict[str, Any]):
    """
    Display key metrics in a clean layout.
    
    Args:
        metrics: Dictionary containing key metrics
    """
    # Create columns for different metric groups
    error_cols = st.columns(3)
    
    # Error metrics
    metric_configs = [
        {"key": "rmse", "label": "RMSE", "format": "{:.4f}", "help": "Root Mean Squared Error - measures the standard deviation of prediction errors"},
        {"key": "mae", "label": "MAE", "format": "{:.4f}", "help": "Mean Absolute Error - average absolute difference between predicted and actual values"},
        {"key": "r2", "label": "R²", "format": "{:.4f}", "help": "Coefficient of determination - proportion of variance explained by the model"}
    ]
    
    for i, config in enumerate(metric_configs):
        if config["key"] in metrics:
            with error_cols[i]:
                value = metrics[config["key"]]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    formatted_value = config["format"].format(value)
                else:
                    formatted_value = "N/A"
                
                st.metric(
                    label=config["label"],
                    value=formatted_value,
                    help=config["help"]
                )
    
    # Additional metrics in expandable section
    with st.expander("Additional Metrics", expanded=False):
        # Create columns for timing metrics
        timing_cols = st.columns(3)
        
        timing_configs = [
            {"key": "training_time", "label": "Training Time", "format": "{:.2f}s", "help": "Time taken to train the MoE model"},
            {"key": "inference_time", "label": "Inference Time", "format": "{:.4f}s", "help": "Average time to generate a prediction"},
            {"key": "total_processing_time", "label": "Total Time", "format": "{:.2f}s", "help": "Total processing time including data preparation and evaluation"}
        ]
        
        for i, config in enumerate(timing_configs):
            if config["key"] in metrics:
                with timing_cols[i]:
                    value = metrics[config["key"]]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        formatted_value = config["format"].format(value)
                    else:
                        formatted_value = "N/A"
                    
                    st.metric(
                        label=config["label"],
                        value=formatted_value,
                        help=config["help"]
                    )
        
        # Sample count metrics
        sample_cols = st.columns(3)
        
        sample_configs = [
            {"key": "sample_count", "label": "Total Samples", "format": "{:,}", "help": "Total number of samples used"},
            {"key": "training_samples", "label": "Training Samples", "format": "{:,}", "help": "Number of samples used for training"},
            {"key": "test_samples", "label": "Test Samples", "format": "{:,}", "help": "Number of samples used for testing"}
        ]
        
        for i, config in enumerate(sample_configs):
            if config["key"] in metrics:
                with sample_cols[i]:
                    value = metrics[config["key"]]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        formatted_value = config["format"].format(value)
                    else:
                        formatted_value = "N/A"
                    
                    st.metric(
                        label=config["label"],
                        value=formatted_value,
                        help=config["help"]
                    )
        
        # Additional error metrics
        additional_cols = st.columns(3)
        
        additional_configs = [
            {"key": "mse", "label": "MSE", "format": "{:.4f}", "help": "Mean Squared Error"},
            {"key": "adjusted_r2", "label": "Adjusted R²", "format": "{:.4f}", "help": "R² adjusted for the number of predictors"},
            {"key": "mape", "label": "MAPE", "format": "{:.2f}%", "help": "Mean Absolute Percentage Error"}
        ]
        
        for i, config in enumerate(additional_configs):
            if config["key"] in metrics:
                with additional_cols[i]:
                    value = metrics[config["key"]]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        # Handle percentage formatting for MAPE
                        if config["key"] == "mape" and value < 1:
                            value = value * 100  # Convert to percentage if not already
                        formatted_value = config["format"].format(value)
                    else:
                        formatted_value = "N/A"
                    
                    st.metric(
                        label=config["label"],
                        value=formatted_value,
                        help=config["help"]
                    )

def display_detailed_metrics(detailed_metrics: Dict[str, Any]):
    """
    Display detailed performance metrics.
    
    Args:
        detailed_metrics: Dictionary containing detailed metrics
    """
    # Convert metrics to DataFrame for display
    metrics_data = []
    
    for metric_name, value in detailed_metrics.items():
        # Handle different data types appropriately
        if isinstance(value, (int, float)):
            if np.isnan(value):
                formatted_value = "N/A"
            elif "percentage" in metric_name.lower() or "ratio" in metric_name.lower():
                formatted_value = f"{value:.2f}%"
            elif value < 0.01:
                formatted_value = f"{value:.6f}"
            elif value < 0.1:
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        
        metrics_data.append({
            "Metric": metric_name,
            "Value": formatted_value
        })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("No detailed metrics available")

def display_error_distribution(error_distribution: Dict[str, Any]):
    """
    Display error distribution visualization.
    
    Args:
        error_distribution: Dictionary containing error distribution data
    """
    if "errors" in error_distribution:
        errors = error_distribution["errors"]
        
        if isinstance(errors, list) and errors:
            # Create histogram of errors
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.histplot(errors, kde=True, ax=ax)
            ax.set_title("Error Distribution")
            ax.set_xlabel("Error")
            ax.set_ylabel("Frequency")
            
            # Add vertical line at mean
            mean_error = np.mean(errors)
            ax.axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.4f}')
            
            # Add vertical line at zero for reference
            ax.axvline(0, color='g', linestyle='-', label='Zero Error')
            
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display error statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Error", f"{np.mean(errors):.4f}")
            
            with col2:
                st.metric("Error Std Dev", f"{np.std(errors):.4f}")
            
            with col3:
                st.metric("Max Error", f"{np.max(np.abs(errors)):.4f}")
            
            with col4:
                st.metric("Median Error", f"{np.median(np.abs(errors)):.4f}")
    
    elif "bins" in error_distribution and "counts" in error_distribution:
        bins = error_distribution["bins"]
        counts = error_distribution["counts"]
        
        if isinstance(bins, list) and isinstance(counts, list) and len(bins) > 1 and len(counts) > 0:
            # Create histogram from bins and counts
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.bar(bins[:-1], counts, width=bins[1]-bins[0], align="edge", alpha=0.7)
            ax.set_title("Error Distribution")
            ax.set_xlabel("Error")
            ax.set_ylabel("Frequency")
            
            plt.tight_layout()
            st.pyplot(fig)
    
    else:
        st.info("Error distribution data is not in the expected format")

def display_segmented_performance(segmented_performance: Dict[str, Any]):
    """
    Display performance metrics segmented by category.
    
    Args:
        segmented_performance: Dictionary containing segmented performance data
    """
    # Check if there's any segment data
    has_segments = False
    segment_keys = []
    
    for key, value in segmented_performance.items():
        if isinstance(value, dict) and value:
            has_segments = True
            segment_keys.append(key)
    
    if not has_segments:
        st.info("No segmented performance data available")
        return
    
    # Allow user to select which segment to view
    selected_segment = st.selectbox(
        "Select Segment Category", 
        options=segment_keys,
        help="Choose a category to see performance broken down by segments"
    )
    
    if selected_segment not in segmented_performance:
        st.error(f"Selected segment '{selected_segment}' not found in performance data")
        return
    
    segment_data = segmented_performance[selected_segment]
    if not isinstance(segment_data, dict) or not segment_data:
        st.info(f"No data available for segment '{selected_segment}'")
        return
    
    # Prepare data for visualization
    segments = []
    rmse_values = []
    mae_values = []
    r2_values = []
    
    for segment_name, metrics in segment_data.items():
        if isinstance(metrics, dict):
            segments.append(segment_name)
            rmse_values.append(metrics.get("rmse", np.nan))
            mae_values.append(metrics.get("mae", np.nan))
            r2_values.append(metrics.get("r2", np.nan))
    
    if not segments:
        st.info(f"No valid segment data found for '{selected_segment}'")
        return
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        "Segment": segments,
        "RMSE": rmse_values,
        "MAE": mae_values,
        "R²": r2_values
    })
    
    # Display as table
    st.dataframe(df)
    
    # Create visualization
    st.subheader(f"Performance by {selected_segment}")
    
    # Create metrics comparison charts
    metrics_to_plot = [col for col in ["RMSE", "MAE", "R²"] if not df[col].isnull().all()]
    
    if metrics_to_plot:
        # Melt the DataFrame for easier plotting
        melted_df = pd.melt(
            df, 
            id_vars=["Segment"],
            value_vars=metrics_to_plot,
            var_name="Metric",
            value_name="Value"
        )
        
        # Create the chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use seaborn for grouped bar chart
        sns.barplot(x="Segment", y="Value", hue="Metric", data=melted_df, ax=ax)
        
        ax.set_title(f"Performance Metrics by {selected_segment}")
        ax.set_xlabel(selected_segment)
        ax.set_ylabel("Value")
        
        # Rotate x-axis labels if there are many segments
        if len(segments) > 5:
            plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Identify best and worst performing segments
        if "RMSE" in metrics_to_plot:
            best_segment = df.loc[df["RMSE"].idxmin(), "Segment"]
            worst_segment = df.loc[df["RMSE"].idxmax(), "Segment"]
            
            st.markdown(f"""
            **Performance Insights:**
            - Best performing segment: **{best_segment}** (lowest RMSE)
            - Worst performing segment: **{worst_segment}** (highest RMSE)
            """)
            
            # Calculate variance in performance across segments
            rmse_std = np.std(df["RMSE"].dropna())
            rmse_mean = np.mean(df["RMSE"].dropna())
            coefficient_of_variation = (rmse_std / rmse_mean) if rmse_mean > 0 else 0
            
            if coefficient_of_variation > 0.5:
                st.warning(
                    f"High performance variability across {selected_segment}s (CV: {coefficient_of_variation:.2f}). "
                    f"This suggests the model performs inconsistently across different {selected_segment}s."
                )
            elif coefficient_of_variation > 0.2:
                st.info(
                    f"Moderate performance variability across {selected_segment}s (CV: {coefficient_of_variation:.2f}). "
                    f"Consider examining factors affecting performance in lower-performing {selected_segment}s."
                )
            else:
                st.success(
                    f"Low performance variability across {selected_segment}s (CV: {coefficient_of_variation:.2f}). "
                    f"The model performs consistently across different {selected_segment}s."
                )
