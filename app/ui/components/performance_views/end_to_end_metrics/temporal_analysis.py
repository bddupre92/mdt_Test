"""
Temporal Analysis Component for End-to-End Performance Metrics

This module provides visualization and analysis of performance metrics over time,
including trend detection, seasonality analysis, and concept drift monitoring.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

logger = logging.getLogger(__name__)

from app.ui.components.performance_views.helpers import safe_get_metric

def render_temporal_analysis(end_to_end_metrics: Dict[str, Any]):
    """
    Render the temporal analysis component for end-to-end performance metrics.
    
    Args:
        end_to_end_metrics: Dictionary containing end-to-end performance metrics
    """
    st.subheader("Temporal Performance Analysis")
    
    # Check if temporal data is available
    temporal_data = safe_get_metric(end_to_end_metrics, "temporal_analysis", {})
    
    if not temporal_data or not isinstance(temporal_data, dict):
        st.info("No temporal analysis data available")
        
        st.markdown("""
        ### Getting Started with Temporal Analysis
        
        To generate temporal analysis data:
        
        1. Include timestamps with your predictions
        2. Use the MoEMetricsCalculator with time-based evaluation
        3. Results will be saved in the temporal_analysis section
        
        Example code:
        ```python
        from baseline_comparison.moe_metrics import MoEMetricsCalculator
        
        calculator = MoEMetricsCalculator()
        metrics = calculator.compute_all_metrics(
            y_true=y_test,
            y_pred=predictions,
            timestamps=test_timestamps  # Include timestamps here
        )
        
        # Update the system state
        system_state.performance_metrics["end_to_end_metrics"]["temporal_analysis"] = metrics["temporal_analysis"]
        state_manager.save_state(system_state, "path/to/checkpoint")
        ```
        """)
        return
    
    # Create tabs for different temporal analyses
    temporal_tabs = st.tabs([
        "Performance Trends",
        "Seasonal Patterns",
        "Concept Drift",
        "Anomaly Detection"
    ])
    
    with temporal_tabs[0]:
        render_performance_trends(temporal_data)
    
    with temporal_tabs[1]:
        render_seasonal_patterns(temporal_data)
    
    with temporal_tabs[2]:
        render_concept_drift(temporal_data)
    
    with temporal_tabs[3]:
        render_anomaly_detection(temporal_data)

def render_performance_trends(temporal_data: Dict[str, Any]):
    """
    Render performance trends over time.
    
    Args:
        temporal_data: Dictionary containing temporal analysis data
    """
    st.subheader("Performance Trends Over Time")
    
    # Extract time series data
    time_series = temporal_data.get("time_series", {})
    
    if not time_series or not isinstance(time_series, dict):
        st.info("No time series data available for trend analysis")
        return
    
    # Check if we have timestamps and metrics
    timestamps = time_series.get("timestamps", [])
    metrics = time_series.get("metrics", {})
    
    if not timestamps or not metrics:
        st.info("Incomplete time series data: missing timestamps or metrics")
        return
    
    # Convert timestamps to datetime if they're strings
    if timestamps and isinstance(timestamps[0], str):
        try:
            timestamps = [datetime.fromisoformat(ts) if 'T' in ts else datetime.strptime(ts, "%Y-%m-%d") 
                          for ts in timestamps]
        except ValueError:
            st.error("Failed to parse timestamp strings")
            return
    
    # Allow user to select metric to visualize
    available_metrics = list(metrics.keys())
    if not available_metrics:
        st.info("No metrics available in time series data")
        return
    
    selected_metric = st.selectbox(
        "Select Metric to Visualize", 
        options=available_metrics,
        index=available_metrics.index("rmse") if "rmse" in available_metrics else 0
    )
    
    if selected_metric not in metrics:
        st.error(f"Selected metric '{selected_metric}' not found in time series data")
        return
    
    metric_values = metrics[selected_metric]
    
    if len(timestamps) != len(metric_values):
        st.error(f"Timestamp count ({len(timestamps)}) doesn't match metric value count ({len(metric_values)})")
        return
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        "Timestamp": timestamps,
        selected_metric: metric_values
    })
    
    # Display time range selector
    time_range = st.slider(
        "Select Time Period",
        min_value=0,
        max_value=len(df) - 1,
        value=(0, len(df) - 1),
        step=1
    )
    
    filtered_df = df.iloc[time_range[0]:time_range[1]+1]
    
    # Create time series plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(x="Timestamp", y=selected_metric, data=filtered_df, ax=ax)
    
    # Add trend line
    if len(filtered_df) > 1:
        x = np.arange(len(filtered_df))
        y = filtered_df[selected_metric].values
        
        # Calculate trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        # Plot trend line
        ax.plot(filtered_df["Timestamp"], p(x), "r--", label=f"Trend: {z[0]:.6f}x + {z[1]:.6f}")
    
    ax.set_title(f"{selected_metric.upper()} Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel(selected_metric.upper())
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    
    st.pyplot(fig)
    
    # Display trend analysis
    if len(filtered_df) > 1:
        trend_slope = z[0]
        
        if selected_metric.lower() in ["rmse", "mae", "mse", "error"]:
            # For error metrics, lower is better
            if trend_slope < -0.001:
                st.success(f"🔽 Improving trend: {selected_metric.upper()} is decreasing by {abs(trend_slope):.6f} per time unit")
            elif trend_slope > 0.001:
                st.error(f"🔼 Concerning trend: {selected_metric.upper()} is increasing by {trend_slope:.6f} per time unit")
            else:
                st.info(f"➡️ Stable trend: {selected_metric.upper()} shows minimal change over time")
        elif selected_metric.lower() in ["r2", "accuracy", "precision", "recall", "f1"]:
            # For these metrics, higher is better
            if trend_slope > 0.001:
                st.success(f"🔼 Improving trend: {selected_metric.upper()} is increasing by {trend_slope:.6f} per time unit")
            elif trend_slope < -0.001:
                st.error(f"🔽 Concerning trend: {selected_metric.upper()} is decreasing by {abs(trend_slope):.6f} per time unit")
            else:
                st.info(f"➡️ Stable trend: {selected_metric.upper()} shows minimal change over time")
        else:
            # For unknown metrics, just report the trend
            st.info(f"Trend slope: {trend_slope:.6f} per time unit")
        
        # Calculate additional statistics
        mean_value = filtered_df[selected_metric].mean()
        std_value = filtered_df[selected_metric].std()
        
        st.markdown(f"""
        **Time Series Statistics:**
        - Mean: {mean_value:.4f}
        - Standard Deviation: {std_value:.4f}
        - Coefficient of Variation: {(std_value / abs(mean_value)) * 100:.2f}%
        """)

def render_seasonal_patterns(temporal_data: Dict[str, Any]):
    """
    Render seasonal patterns analysis.
    
    Args:
        temporal_data: Dictionary containing temporal analysis data
    """
    st.subheader("Seasonal Performance Patterns")
    
    # Extract seasonal analysis data
    seasonal_analysis = temporal_data.get("seasonal_analysis", {})
    
    if not seasonal_analysis or not isinstance(seasonal_analysis, dict):
        st.info("No seasonal analysis data available")
        return
    
    # Check what types of seasonal patterns are available
    available_patterns = []
    for pattern_type in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
        if pattern_type in seasonal_analysis:
            available_patterns.append(pattern_type)
    
    if not available_patterns:
        st.info("No pattern data available in seasonal analysis")
        return
    
    # Allow user to select pattern type
    selected_pattern = st.selectbox(
        "Select Seasonal Pattern", 
        options=available_patterns
    )
    
    if selected_pattern not in seasonal_analysis:
        st.error(f"Selected pattern '{selected_pattern}' not found in seasonal analysis")
        return
    
    pattern_data = seasonal_analysis[selected_pattern]
    
    if not isinstance(pattern_data, dict):
        st.info(f"Invalid pattern data for '{selected_pattern}'")
        return
    
    # Get available metrics for the selected pattern
    available_metrics = list(pattern_data.keys())
    
    if not available_metrics:
        st.info(f"No metrics available for '{selected_pattern}' pattern")
        return
    
    # Allow user to select metric
    selected_metric = st.selectbox(
        "Select Metric", 
        options=available_metrics,
        index=available_metrics.index("rmse") if "rmse" in available_metrics else 0
    )
    
    if selected_metric not in pattern_data:
        st.error(f"Selected metric '{selected_metric}' not found in pattern data")
        return
    
    metric_pattern = pattern_data[selected_metric]
    
    if not isinstance(metric_pattern, dict) or "periods" not in metric_pattern or "values" not in metric_pattern:
        st.info(f"Invalid pattern data for '{selected_metric}'")
        return
    
    periods = metric_pattern["periods"]
    values = metric_pattern["values"]
    
    if len(periods) != len(values):
        st.error(f"Period count ({len(periods)}) doesn't match value count ({len(values)})")
        return
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        "Period": periods,
        selected_metric: values
    })
    
    # Create seasonal pattern plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(df["Period"], df[selected_metric], width=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', rotation=0
        )
    
    ax.set_title(f"{selected_metric.upper()} by {selected_pattern.capitalize()} Period")
    ax.set_xlabel(f"{selected_pattern.capitalize()} Period")
    ax.set_ylabel(selected_metric.upper())
    
    # Rotate x-axis labels if there are many periods
    if len(periods) > 10:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display pattern analysis
    if len(df) > 1:
        max_period = df.loc[df[selected_metric].idxmax(), "Period"]
        min_period = df.loc[df[selected_metric].idxmin(), "Period"]
        
        if selected_metric.lower() in ["rmse", "mae", "mse", "error"]:
            # For error metrics, lower is better
            st.markdown(f"""
            **Pattern Analysis:**
            - Best performance in period: **{min_period}** (lowest {selected_metric.upper()})
            - Worst performance in period: **{max_period}** (highest {selected_metric.upper()})
            """)
        elif selected_metric.lower() in ["r2", "accuracy", "precision", "recall", "f1"]:
            # For these metrics, higher is better
            st.markdown(f"""
            **Pattern Analysis:**
            - Best performance in period: **{max_period}** (highest {selected_metric.upper()})
            - Worst performance in period: **{min_period}** (lowest {selected_metric.upper()})
            """)
        else:
            # For unknown metrics
            st.markdown(f"""
            **Pattern Analysis:**
            - Highest {selected_metric.upper()} in period: **{max_period}**
            - Lowest {selected_metric.upper()} in period: **{min_period}**
            """)
        
        # Calculate variation
        max_value = df[selected_metric].max()
        min_value = df[selected_metric].min()
        mean_value = df[selected_metric].mean()
        
        variation = (max_value - min_value) / mean_value if mean_value != 0 else 0
        
        if variation > 0.5:
            st.warning(f"High seasonal variation ({variation:.2f}): Performance varies significantly across {selected_pattern} periods")
        elif variation > 0.2:
            st.info(f"Moderate seasonal variation ({variation:.2f}): Some performance differences across {selected_pattern} periods")
        else:
            st.success(f"Low seasonal variation ({variation:.2f}): Consistent performance across {selected_pattern} periods")

def render_concept_drift(temporal_data: Dict[str, Any]):
    """
    Render concept drift analysis.
    
    Args:
        temporal_data: Dictionary containing temporal analysis data
    """
    st.subheader("Concept Drift Analysis")
    
    # Extract concept drift data
    concept_drift = temporal_data.get("concept_drift", {})
    
    if not concept_drift or not isinstance(concept_drift, dict):
        st.info("No concept drift analysis data available")
        
        st.markdown("""
        **What is Concept Drift?**
        
        Concept drift refers to changes in the statistical properties of the target variable over time,
        which can lead to declining model performance. Detecting concept drift is crucial for maintaining
        model effectiveness in production environments.
        
        To enable concept drift analysis, make sure to:
        1. Include timestamps with your predictions
        2. Enable drift detection in your metrics calculation
        3. Periodically retrain your model with fresh data
        """)
        return
    
    # Display concept drift metrics
    drift_detected = concept_drift.get("drift_detected", False)
    drift_score = concept_drift.get("drift_score", 0)
    detection_method = concept_drift.get("detection_method", "Unknown")
    
    # Create a visual indicator for drift severity
    severity_color = "green"
    severity_text = "No significant drift detected"
    
    if drift_score > 0.7:
        severity_color = "red"
        severity_text = "Severe concept drift detected"
    elif drift_score > 0.3:
        severity_color = "orange"
        severity_text = "Moderate concept drift detected"
    elif drift_score > 0.1:
        severity_color = "yellow"
        severity_text = "Mild concept drift detected"
    
    # Create a progress bar to visualize drift score
    st.markdown(f"<h3 style='color: {severity_color};'>{severity_text}</h3>", unsafe_allow_html=True)
    st.progress(float(drift_score))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Drift Score", 
            f"{drift_score:.4f}",
            help="Measure of concept drift severity from 0 (no drift) to 1 (severe drift)"
        )
    
    with col2:
        st.metric(
            "Drift Detected", 
            "Yes" if drift_detected else "No",
            help="Whether concept drift has been automatically detected"
        )
    
    with col3:
        st.metric(
            "Detection Method", 
            detection_method,
            help="Statistical method used for drift detection"
        )
    
    # Display change points if available
    change_points = concept_drift.get("change_points", [])
    
    if change_points and isinstance(change_points, list):
        st.subheader("Detected Change Points")
        
        change_point_data = []
        for cp in change_points:
            if isinstance(cp, dict):
                change_point_data.append({
                    "Timestamp": cp.get("timestamp", "Unknown"),
                    "Score": cp.get("score", 0),
                    "Confidence": cp.get("confidence", 0),
                    "Type": cp.get("type", "Unknown")
                })
        
        if change_point_data:
            change_df = pd.DataFrame(change_point_data)
            st.dataframe(change_df)
            
            # Create visualization of change points
            if "time_series" in temporal_data and isinstance(temporal_data["time_series"], dict):
                time_series = temporal_data["time_series"]
                
                if "timestamps" in time_series and "metrics" in time_series:
                    timestamps = time_series["timestamps"]
                    metrics = time_series["metrics"]
                    
                    # Choose a metric to visualize
                    metric_key = next((key for key in ["rmse", "mae", "error"] if key in metrics), next(iter(metrics)))
                    
                    if metric_key in metrics:
                        metric_values = metrics[metric_key]
                        
                        if len(timestamps) == len(metric_values):
                            # Create time series with change points highlighted
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Ensure timestamps are in correct format
                            if timestamps and isinstance(timestamps[0], str):
                                try:
                                    timestamps = [datetime.fromisoformat(ts) if 'T' in ts else datetime.strptime(ts, "%Y-%m-%d") 
                                                for ts in timestamps]
                                except ValueError:
                                    # If parsing fails, use indices
                                    timestamps = list(range(len(timestamps)))
                            
                            # Plot the time series
                            ax.plot(timestamps, metric_values, label=metric_key.upper())
                            
                            # Mark change points
                            for cp in change_points:
                                if isinstance(cp, dict) and "timestamp" in cp:
                                    cp_timestamp = cp["timestamp"]
                                    
                                    # Convert to same format as main timestamps if needed
                                    if isinstance(cp_timestamp, str) and timestamps and not isinstance(timestamps[0], str):
                                        try:
                                            cp_timestamp = datetime.fromisoformat(cp_timestamp) if 'T' in cp_timestamp else datetime.strptime(cp_timestamp, "%Y-%m-%d")
                                        except ValueError:
                                            # If parsing fails, skip this change point
                                            continue
                                    
                                    # Find closest timestamp
                                    if cp_timestamp in timestamps:
                                        idx = timestamps.index(cp_timestamp)
                                    else:
                                        # Skip this change point if not found
                                        continue
                                    
                                    # Mark the change point
                                    if 0 <= idx < len(metric_values):
                                        ax.axvline(timestamps[idx], color='r', linestyle='--', alpha=0.7)
                                        ax.text(
                                            timestamps[idx], 
                                            min(metric_values) - (max(metric_values) - min(metric_values)) * 0.05,
                                            f"Change Point",
                                            rotation=90,
                                            verticalalignment='bottom'
                                        )
                            
                            ax.set_title(f"{metric_key.upper()} with Detected Change Points")
                            ax.set_xlabel("Time")
                            ax.set_ylabel(metric_key.upper())
                            
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.legend()
                            
                            st.pyplot(fig)
    
    # Display drift cause analysis if available
    drift_causes = concept_drift.get("drift_causes", [])
    
    if drift_causes and isinstance(drift_causes, list):
        st.subheader("Potential Drift Causes")
        
        for cause in drift_causes:
            if isinstance(cause, dict):
                cause_type = cause.get("type", "Unknown")
                cause_score = cause.get("score", 0)
                cause_description = cause.get("description", "No description available")
                
                st.markdown(f"""
                ### {cause_type} (Score: {cause_score:.2f})
                
                {cause_description}
                """)
    
    # Display recommendations if available
    recommendations = concept_drift.get("recommendations", [])
    
    if recommendations and isinstance(recommendations, list):
        st.subheader("Recommendations")
        
        for rec in recommendations:
            if isinstance(rec, str):
                st.markdown(f"- {rec}")
            elif isinstance(rec, dict) and "text" in rec:
                priority = rec.get("priority", "medium")
                priority_color = {
                    "high": "red",
                    "medium": "orange",
                    "low": "blue"
                }.get(priority.lower(), "black")
                
                st.markdown(f"- <span style='color: {priority_color};'>[{priority.upper()}]</span> {rec['text']}", unsafe_allow_html=True)

def render_anomaly_detection(temporal_data: Dict[str, Any]):
    """
    Render anomaly detection in temporal performance.
    
    Args:
        temporal_data: Dictionary containing temporal analysis data
    """
    st.subheader("Performance Anomaly Detection")
    
    # Extract anomaly detection data
    anomalies = temporal_data.get("anomalies", {})
    
    if not anomalies or not isinstance(anomalies, dict):
        st.info("No anomaly detection data available")
        return
    
    # Get available metrics with anomaly detection
    available_metrics = []
    for metric, anomaly_data in anomalies.items():
        if isinstance(anomaly_data, dict) and anomaly_data:
            available_metrics.append(metric)
    
    if not available_metrics:
        st.info("No metrics with anomaly detection available")
        return
    
    # Allow user to select metric
    selected_metric = st.selectbox(
        "Select Metric", 
        options=available_metrics,
        index=available_metrics.index("rmse") if "rmse" in available_metrics else 0
    )
    
    if selected_metric not in anomalies:
        st.error(f"Selected metric '{selected_metric}' not found in anomaly data")
        return
    
    anomaly_data = anomalies[selected_metric]
    
    if not isinstance(anomaly_data, dict):
        st.info(f"Invalid anomaly data for '{selected_metric}'")
        return
    
    # Display summary statistics
    anomaly_count = anomaly_data.get("anomaly_count", 0)
    anomaly_percentage = anomaly_data.get("anomaly_percentage", 0)
    detection_method = anomaly_data.get("detection_method", "Unknown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Anomaly Count", 
            f"{anomaly_count}",
            help="Number of detected anomalies in the time series"
        )
    
    with col2:
        st.metric(
            "Anomaly Percentage", 
            f"{anomaly_percentage:.2f}%",
            help="Percentage of time points with anomalous behavior"
        )
    
    with col3:
        st.metric(
            "Detection Method", 
            detection_method,
            help="Method used for anomaly detection"
        )
    
    # Visualize anomalies if time series data is available
    if "time_series" in temporal_data and isinstance(temporal_data["time_series"], dict):
        time_series = temporal_data["time_series"]
        
        if "timestamps" in time_series and "metrics" in time_series:
            timestamps = time_series["timestamps"]
            metrics = time_series["metrics"]
            
            if selected_metric in metrics:
                metric_values = metrics[selected_metric]
                
                if len(timestamps) == len(metric_values):
                    # Check if we have anomaly flags
                    anomaly_flags = anomaly_data.get("anomaly_flags", [])
                    
                    if len(anomaly_flags) == len(timestamps):
                        # Create time series with anomalies highlighted
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Ensure timestamps are in correct format
                        if timestamps and isinstance(timestamps[0], str):
                            try:
                                timestamps = [datetime.fromisoformat(ts) if 'T' in ts else datetime.strptime(ts, "%Y-%m-%d") 
                                            for ts in timestamps]
                            except ValueError:
                                # If parsing fails, use indices
                                timestamps = list(range(len(timestamps)))
                        
                        # Create a DataFrame with the data
                        df = pd.DataFrame({
                            "Timestamp": timestamps,
                            "Value": metric_values,
                            "Anomaly": anomaly_flags
                        })
                        
                        # Plot normal points
                        normal_df = df[~df["Anomaly"]]
                        ax.plot(normal_df["Timestamp"], normal_df["Value"], 'b-', label='Normal')
                        
                        # Plot anomalous points
                        anomaly_df = df[df["Anomaly"]]
                        ax.scatter(anomaly_df["Timestamp"], anomaly_df["Value"], color='red', s=50, label='Anomaly')
                        
                        ax.set_title(f"{selected_metric.upper()} with Detected Anomalies")
                        ax.set_xlabel("Time")
                        ax.set_ylabel(selected_metric.upper())
                        
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.legend()
                        
                        st.pyplot(fig)
                        
                        # Display anomalies table if there are any
                        if not anomaly_df.empty:
                            st.subheader("Detected Anomalies")
                            
                            # Create a clean DataFrame for display
                            display_df = anomaly_df.copy()
                            display_df["Value"] = display_df["Value"].apply(lambda x: f"{x:.4f}")
                            
                            st.dataframe(display_df[["Timestamp", "Value"]])
                        
                        # Display anomaly patterns if available
                        patterns = anomaly_data.get("patterns", [])
                        
                        if patterns and isinstance(patterns, list):
                            st.subheader("Anomaly Patterns")
                            
                            for i, pattern in enumerate(patterns):
                                if isinstance(pattern, dict):
                                    pattern_type = pattern.get("type", "Unknown")
                                    pattern_score = pattern.get("score", 0)
                                    pattern_description = pattern.get("description", "No description available")
                                    
                                    st.markdown(f"""
                                    #### Pattern {i+1}: {pattern_type} (Score: {pattern_score:.2f})
                                    
                                    {pattern_description}
                                    """)
                    else:
                        # If we don't have anomaly flags, plot the raw time series
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        ax.plot(timestamps, metric_values)
                        
                        ax.set_title(f"{selected_metric.upper()} Time Series")
                        ax.set_xlabel("Time")
                        ax.set_ylabel(selected_metric.upper())
                        
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        st.info("Anomaly flags not available for visualization")
    
    # Display most significant anomalies if available
    significant_anomalies = anomaly_data.get("significant_anomalies", [])
    
    if significant_anomalies and isinstance(significant_anomalies, list):
        st.subheader("Most Significant Anomalies")
        
        for anomaly in significant_anomalies:
            if isinstance(anomaly, dict):
                timestamp = anomaly.get("timestamp", "Unknown")
                severity = anomaly.get("severity", 0)
                value = anomaly.get("value", "N/A")
                expected_value = anomaly.get("expected_value", "N/A")
                deviation = anomaly.get("deviation", "N/A")
                
                # Format values for display
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
                
                if isinstance(expected_value, (int, float)):
                    expected_value = f"{expected_value:.4f}"
                
                if isinstance(deviation, (int, float)):
                    deviation = f"{deviation:.4f}"
                
                # Display with severity-based formatting
                severity_color = "blue"
                if severity >= 0.8:
                    severity_color = "red"
                elif severity >= 0.5:
                    severity_color = "orange"
                
                st.markdown(f"""
                <div style="border-left: 4px solid {severity_color}; padding-left: 10px; margin-bottom: 15px;">
                    <p><strong>Timestamp:</strong> {timestamp}</p>
                    <p><strong>Severity:</strong> {severity:.2f}</p>
                    <p><strong>Actual Value:</strong> {value} | <strong>Expected Value:</strong> {expected_value}</p>
                    <p><strong>Deviation:</strong> {deviation}</p>
                </div>
                """, unsafe_allow_html=True)
