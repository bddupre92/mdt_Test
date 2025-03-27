"""
Performance Analysis UI Components

This module provides UI components for the Performance Analysis dashboard,
enabling detailed analysis of MoE system performance through metrics visualization,
statistical tests, and comparison with baseline methods.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from pathlib import Path
import json
from typing import Dict, Any, List, Optional

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

logger = logging.getLogger(__name__)

# Import components for specific analysis views
from app.ui.components.performance_views.expert_benchmarks import render_expert_benchmarks
from app.ui.components.performance_views.gating_analysis import render_gating_analysis
from app.ui.components.performance_views.end_to_end_metrics import render_end_to_end_metrics

# Import the new performance analysis module
from app.ui.components.performance_views.performance_analysis import render_performance_analysis
from moe_framework.interfaces.base import SystemState

def render_performance_analysis_ui(active_tab: str, performance_data: Dict[str, Any]):
    """
    Render the Performance Analysis UI based on the selected tab.
    This component is designed to adapt to various data formats and structures.
    
    Args:
        active_tab: The currently selected analysis tab
        performance_data: Dictionary containing performance metrics and analysis results
    """
    if not performance_data:
        st.warning("No performance data available")
        return
    
    # Dynamic section for data structure inspection and debugging
    with st.sidebar.expander("Data Structure Inspector", expanded=False):
        st.markdown("### Data Format Overview")
        st.markdown("Use this section to understand the structure of your performance data.")
        
        # Show top-level keys
        top_level_keys = list(performance_data.keys())
        st.write("Top-level keys:", top_level_keys)
        
        # Allow inspection of specific keys
        if top_level_keys:
            inspect_key = st.selectbox("Inspect key:", ["None"] + top_level_keys)
            if inspect_key != "None":
                st.json(performance_data.get(inspect_key, {}))
    
    # Display a note about accessing preprocessing tools
    with st.sidebar.expander("Data Preprocessing", expanded=False):
        st.markdown("""To configure data preprocessing, please use the Data Configuration Dashboard.
        There you can access the drag-and-drop pipeline builder for creating preprocessing workflows.""")
        if st.button("Go to Data Configuration"):
            # This would be handled by Streamlit's navigation
            st.info("Navigation button clicked. Actual navigation would be handled by the application's routing.")
    
    # Smart data detection and extraction for each component
    # This allows flexibility in data format without requiring specific structures
    
    # Helper function to find keys in the data by partial matching
    def find_keys(data, patterns):
        """Find keys in the data that match any of the patterns."""
        if not isinstance(data, dict):
            return {}
            
        results = {}
        for key, value in data.items():
            key_lower = key.lower()
            for pattern in patterns:
                if pattern in key_lower:
                    results[key] = value
                    break
        return results
    
    # Display tab content based on selection
    if active_tab == "Overview":
        render_overview(performance_data)
    elif active_tab == "Expert Benchmarks":
        # Try to find expert-related data anywhere in the structure
        expert_patterns = ["expert", "benchmark", "specialist", "model_performance"]
        expert_data = performance_data.get("expert_benchmarks", {})
        
        # If expert_benchmarks key doesn't exist, try to find it elsewhere
        if not expert_data:
            expert_data = find_keys(performance_data, expert_patterns)
            # If still not found, look one level deeper
            if not expert_data:
                for key, value in performance_data.items():
                    if isinstance(value, dict):
                        nested_experts = find_keys(value, expert_patterns)
                        if nested_experts:
                            expert_data = nested_experts
                            break
        
        render_expert_benchmarks(expert_data)
    elif active_tab == "Gating Network Analysis":
        # Try to find gating-related data anywhere in the structure
        gating_patterns = ["gating", "router", "selection", "weight", "allocation"]
        gating_data = performance_data.get("gating_evaluation", {})
        
        # If gating_evaluation key doesn't exist, try to find it elsewhere
        if not gating_data:
            gating_data = find_keys(performance_data, gating_patterns)
            # If still not found, look one level deeper
            if not gating_data:
                for key, value in performance_data.items():
                    if isinstance(value, dict):
                        nested_gating = find_keys(value, gating_patterns)
                        if nested_gating:
                            gating_data = nested_gating
                            break
        
        render_gating_analysis(gating_data)
    elif active_tab == "End-to-End Performance":
        # Use our comprehensive end-to-end metrics module
        system_state = create_temp_system_state(performance_data)
        # This will handle the data structure internally
        render_performance_analysis(system_state)
    elif active_tab == "Baseline Comparisons":
        # Try to find comparison-related data anywhere in the structure
        comparison_patterns = ["comparison", "baseline", "model", "benchmark"]
        comparison_data = performance_data.get("baseline_comparisons", 
                                             performance_data.get("end_to_end_metrics", {}))
        
        # If neither key exists, try to find it elsewhere
        if not comparison_data or len(comparison_data) == 0:
            comparison_data = find_keys(performance_data, comparison_patterns)
            # If still not found, look one level deeper
            if not comparison_data:
                for key, value in performance_data.items():
                    if isinstance(value, dict):
                        nested_comparisons = find_keys(value, comparison_patterns)
                        if nested_comparisons:
                            comparison_data = nested_comparisons
                            break
        
        # Adapt end_to_end_metrics renderer for baseline comparisons
        render_end_to_end_metrics(comparison_data)
    elif active_tab == "Statistical Tests":
        # Try to find statistical test data anywhere in the structure
        stats_patterns = ["statistic", "test", "significance", "p_value", "hypothesis"]
        stats_data = performance_data.get("statistical_tests", {})
        
        # If statistical_tests key doesn't exist, try to find it elsewhere
        if not stats_data:
            stats_data = find_keys(performance_data, stats_patterns)
            # If still not found, look one level deeper
            if not stats_data:
                for key, value in performance_data.items():
                    if isinstance(value, dict):
                        nested_stats = find_keys(value, stats_patterns)
                        if nested_stats:
                            stats_data = nested_stats
                            break
        
        # If we still don't have statistical data, use end-to-end metrics for now
        if not stats_data:
            stats_data = performance_data.get("end_to_end_metrics", {})
            
        render_end_to_end_metrics(stats_data)
    elif active_tab == "Visualizations":
        st.subheader("Performance Visualizations")
        st.markdown("Visualizations are now integrated into each respective analysis tab.")
        st.info("Navigate to the specific analysis sections to view related visualizations.")
        
        # Offer a unified visualization view option
        if st.checkbox("Show unified visualization dashboard"):
            # Try to extract all potentially visualizable data
            system_state = create_temp_system_state(performance_data)
            render_performance_analysis(system_state, show_all=True)
    else:
        st.error(f"Unknown tab: {active_tab}")

def render_overview(performance_data: Dict[str, Any]):
    """
    Render the overview tab with summary of all performance metrics.
    
    Args:
        performance_data: Dictionary containing all performance metrics
    """
    st.header("Performance Analysis Overview")
    
    st.markdown("""
    This dashboard provides comprehensive analysis of your MoE system's performance.
    Use the sidebar to navigate to specific analysis areas.
    """)
    
    # Check if performance metrics exist
    if not performance_data or all(not v for v in performance_data.values() if isinstance(v, dict)):
        st.warning("No performance metrics have been calculated yet")
        
        st.markdown("""
        ### Getting Started
        
        To generate performance metrics:
        
        1. Run a complete evaluation of your MoE system using the Performance Analyzer
        2. The system will generate metrics for expert models, gating network, and end-to-end performance
        3. Results will be saved to the system state and can be loaded here
        
        You can also use the API to programmatically analyze performance:
        
        ```python
        from moe_framework.analysis.performance_analyzer import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.analyze_moe_performance(moe_system, test_data)
        
        # Save results to the system state
        system_state.performance_metrics = metrics
        state_manager.save_state(system_state, "path/to/checkpoint")
        ```
        """)
        return
    
    # Display experiment info if available
    if "experiment_id" in performance_data and performance_data["experiment_id"]:
        st.markdown(f"**Experiment ID**: `{performance_data['experiment_id']}`")
    
    # Create tabs for different overview sections
    overview_tabs = st.tabs([
        "Key Metrics", 
        "Performance Summary", 
        "Expert Comparison",
        "Data Usage"
    ])
    
    with overview_tabs[0]:
        render_key_metrics(performance_data)
    
    with overview_tabs[1]:
        render_performance_summary(performance_data)
    
    with overview_tabs[2]:
        render_expert_comparison_summary(performance_data)
    
    with overview_tabs[3]:
        render_data_usage_summary(performance_data)

def render_key_metrics(performance_data: Dict[str, Any]):
    """
    Render the key metrics section of the overview.
    
    Args:
        performance_data: Dictionary containing all performance metrics
    """
    st.subheader("Key Performance Metrics")
    
    # Extract end-to-end metrics if available
    end_to_end = performance_data.get("end_to_end_metrics", {})
    
    if not end_to_end:
        st.info("No end-to-end metrics available")
        return
    
    # Create a 2x2 grid for key metrics
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        metric_value = end_to_end.get("rmse", "N/A")
        st.metric(
            label="RMSE", 
            value=f"{metric_value:.4f}" if isinstance(metric_value, (int, float)) else metric_value
        )
    
    with col2:
        metric_value = end_to_end.get("mae", "N/A")
        st.metric(
            label="MAE", 
            value=f"{metric_value:.4f}" if isinstance(metric_value, (int, float)) else metric_value
        )
    
    with col3:
        metric_value = end_to_end.get("r2", "N/A")
        st.metric(
            label="R²", 
            value=f"{metric_value:.4f}" if isinstance(metric_value, (int, float)) else metric_value
        )
    
    with col4:
        # Get optimal expert selection rate if available
        gating = performance_data.get("gating_evaluation", {})
        metric_value = gating.get("optimal_expert_selection_rate", "N/A")
        st.metric(
            label="Optimal Expert Rate", 
            value=f"{metric_value:.2f}%" if isinstance(metric_value, (int, float)) else metric_value
        )

def render_performance_summary(performance_data: Dict[str, Any]):
    """
    Render the performance summary section of the overview.
    
    Args:
        performance_data: Dictionary containing all performance metrics
    """
    st.subheader("Performance Summary")
    
    # Extract relevant metrics
    end_to_end = performance_data.get("end_to_end_metrics", {})
    baseline = performance_data.get("baseline_comparisons", {})
    
    if not end_to_end and not baseline:
        st.info("No performance summary data available")
        return
    
    # Create summary data for comparison
    summary_data = {}
    
    # Add MoE system performance
    if end_to_end:
        summary_data["MoE System"] = {
            "RMSE": end_to_end.get("rmse", np.nan),
            "MAE": end_to_end.get("mae", np.nan),
            "R²": end_to_end.get("r2", np.nan)
        }
    
    # Add baseline performances
    if baseline and "baseline_metrics" in baseline:
        for method_name, metrics in baseline.get("baseline_metrics", {}).items():
            summary_data[method_name] = {
                "RMSE": metrics.get("rmse", np.nan),
                "MAE": metrics.get("mae", np.nan),
                "R²": metrics.get("r2", np.nan)
            }
    
    # Create a DataFrame for display
    if summary_data:
        df = pd.DataFrame(summary_data).T
        
        # Format the dataframe for display
        formatted_df = df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A")
        
        # Highlight the best performance
        st.dataframe(formatted_df)
        
        # Visualize the comparison
        if len(summary_data) > 1:
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot RMSE
            metrics_to_plot = ["RMSE", "MAE", "R²"]
            for i, metric in enumerate(metrics_to_plot):
                # Extract metric values and method names
                values = [data.get(metric, np.nan) for data in summary_data.values()]
                methods = list(summary_data.keys())
                
                # Create bar chart
                bars = ax[i].bar(methods, values)
                ax[i].set_title(f"{metric} Comparison")
                ax[i].set_ylabel(metric)
                ax[i].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax[i].text(
                            bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}',
                            ha='center', va='bottom', rotation=0
                        )
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("No performance comparison data available")

def render_expert_comparison_summary(performance_data: Dict[str, Any]):
    """
    Render the expert comparison summary section of the overview.
    
    Args:
        performance_data: Dictionary containing all performance metrics
    """
    st.subheader("Expert Model Comparison")
    
    # Extract expert benchmarks
    expert_benchmarks = performance_data.get("expert_benchmarks", {})
    
    if not expert_benchmarks:
        st.info("No expert benchmark data available")
        return
    
    # Create expert performance data
    expert_data = {}
    
    for expert_name, metrics in expert_benchmarks.items():
        if isinstance(metrics, dict):
            expert_data[expert_name] = {
                "RMSE": metrics.get("rmse", np.nan),
                "MAE": metrics.get("mae", np.nan),
                "R²": metrics.get("r2", np.nan)
            }
    
    # Create a DataFrame for display
    if expert_data:
        df = pd.DataFrame(expert_data).T
        
        # Format the dataframe for display
        formatted_df = df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A")
        
        # Display the dataframe
        st.dataframe(formatted_df)
        
        # Visualize the expert comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract RMSE values and expert names
        rmse_values = [data.get("RMSE", np.nan) for data in expert_data.values()]
        expert_names = list(expert_data.keys())
        
        # Create bar chart for RMSE
        bars = ax.bar(expert_names, rmse_values)
        ax.set_title("Expert Model RMSE Comparison")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0
                )
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Extract gating network information if available
        gating_eval = performance_data.get("gating_evaluation", {})
        if gating_eval and "expert_selection_counts" in gating_eval:
            st.subheader("Expert Selection Distribution")
            
            # Extract selection counts
            selection_counts = gating_eval.get("expert_selection_counts", {})
            
            if selection_counts:
                # Create pie chart
                fig, ax = plt.subplots(figsize=(10, 8))
                
                labels = list(selection_counts.keys())
                sizes = list(selection_counts.values())
                
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.info("No expert performance data available")

def create_temp_system_state(performance_data: Dict[str, Any]) -> SystemState:
    """
    Create a temporary SystemState object from performance data dictionary.
    This function is designed to be flexible and handle various data formats
    by intelligently identifying performance metrics regardless of structure.
    
    Args:
        performance_data: Dictionary containing performance metrics in any structure
        
    Returns:
        SystemState object with performance metrics populated
    """
    import copy
    from moe_framework.interfaces.base import SystemState
    
    system_state = SystemState()
    
    # Create a deep copy to avoid modifying the original data
    metrics_data = copy.deepcopy(performance_data)
    
    # If data is in a performance_metrics key, use that directly
    if isinstance(metrics_data, dict) and "performance_metrics" in metrics_data:
        system_state.performance_metrics = metrics_data["performance_metrics"]
        return system_state
    
    # Check if the data contains any of our expected top-level keys directly
    expected_keys = [
        "expert_benchmarks", "gating_evaluation", "end_to_end_metrics",
        "baseline_comparisons", "statistical_tests"
    ]
    
    if isinstance(metrics_data, dict) and any(key in metrics_data for key in expected_keys):
        # Data appears to be in the expected format already
        system_state.performance_metrics = metrics_data
        return system_state
    
    # If metrics_data is not a dict or doesn't match our expected patterns, use as is
    system_state.performance_metrics = metrics_data
    return system_state


def render_data_usage_summary(performance_data: Dict[str, Any]):
    """
    Render the data usage summary section of the overview.
    
    Args:
        performance_data: Dictionary containing all performance metrics
    """
    st.subheader("Data Usage Summary")
    
    # Extract data usage information
    end_to_end = performance_data.get("end_to_end_metrics", {})
    temporal = performance_data.get("temporal_analysis", {})
    
    # Check if we have data count information
    if "data_counts" in end_to_end:
        counts = end_to_end["data_counts"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Samples", counts.get("train", "N/A"))
        
        with col2:
            st.metric("Validation Samples", counts.get("validation", "N/A"))
        
        with col3:
            st.metric("Test Samples", counts.get("test", "N/A"))
    
    # Check if we have temporal distribution information
    if temporal and "temporal_distribution" in temporal:
        st.subheader("Temporal Data Distribution")
        
        # Extract temporal distribution data
        distribution = temporal["temporal_distribution"]
        
        # Create a simple bar chart of data counts by time period
        if isinstance(distribution, dict):
            periods = list(distribution.keys())
            counts = list(distribution.values())
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(periods, counts)
            ax.set_title("Data Distribution by Time Period")
            ax.set_ylabel("Sample Count")
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # If we have neither, show a message
    if not end_to_end.get("data_counts") and not (temporal and "temporal_distribution" in temporal):
        st.info("No data usage information available")
