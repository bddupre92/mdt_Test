"""
Performance Analysis Dashboard

This module provides a comprehensive dashboard for analyzing the performance
of the MoE framework. It integrates various performance views, including:
- Expert Benchmarks
- Gating Analysis
- End-to-End Metrics

The dashboard is designed to be modular and extensible, allowing for easy
addition of new performance metrics and visualizations.
"""

import streamlit as st
from typing import Dict, Any, Optional
import logging

from moe_framework.interfaces.base import SystemState

from .expert_benchmarks import render_expert_benchmarks
from .gating_analysis import render_gating_analysis
from .end_to_end_metrics import render_end_to_end_metrics

logger = logging.getLogger(__name__)

def render_performance_analysis(system_state: SystemState, session_state: Optional[Dict[str, Any]] = None):
    """
    Render the Performance Analysis Dashboard.
    
    Args:
        system_state: The current system state object
        session_state: Optional Streamlit session state for maintaining UI state
    """
    st.title("Performance Analysis Dashboard")
    
    st.markdown("""
    This dashboard provides comprehensive analysis tools for evaluating the performance
    of your Mixture of Experts (MoE) system. Select a view from the tabs below to analyze
    different aspects of the system's performance.
    """)
    
    # Check if we have performance metrics in the system state
    # Handle both object and dictionary access patterns
    has_metrics = False
    performance_metrics = None
    
    if isinstance(system_state, dict):
        performance_metrics = system_state.get('performance_metrics', {})
        has_metrics = bool(performance_metrics)
    else:
        has_metrics = hasattr(system_state, 'performance_metrics') and bool(system_state.performance_metrics)
        if has_metrics:
            performance_metrics = system_state.performance_metrics
            
    if not has_metrics:
        st.warning("No performance metrics found in the system state")
        
        st.markdown("""
        ### Getting Started with Performance Analysis
        
        Performance metrics need to be computed and stored in the system state.
        Follow these steps to generate performance metrics:
        
        1. Run your MoE system on a test dataset
        2. Use the `MoEMetricsCalculator` to compute performance metrics
        3. Update the system state with the calculated metrics
        4. Save the updated system state
        
        ```python
        from baseline_comparison.moe_metrics import MoEMetricsCalculator
        
        # Calculate metrics
        calculator = MoEMetricsCalculator()
        metrics = calculator.compute_all_metrics(
            y_true=y_test,
            y_pred=predictions,
            expert_weights=expert_weights,
            expert_outputs=individual_expert_outputs
        )
        
        # Update system state
        system_state.performance_metrics = {
            "expert_benchmarks": metrics["expert_metrics"],
            "gating_evaluation": metrics["gating_metrics"],
            "end_to_end_metrics": metrics["end_to_end_metrics"]
        }
        
        # Save the state
        state_manager.save_state(system_state, "path/to/checkpoint")
        ```
        """)
        return
    
    # Create tabs for different performance views
    performance_tabs = st.tabs([
        "End-to-End Metrics",
        "Expert Benchmarks",
        "Gating Analysis"
    ])
    
    with performance_tabs[0]:
        # Render end-to-end metrics view
        end_to_end_metrics = performance_metrics.get("end_to_end_metrics", {}) if isinstance(performance_metrics, dict) else getattr(performance_metrics, "end_to_end_metrics", {})
        render_end_to_end_metrics(end_to_end_metrics)
    
    with performance_tabs[1]:
        # Render expert benchmarks view
        expert_benchmarks = performance_metrics.get("expert_benchmarks", {}) if isinstance(performance_metrics, dict) else getattr(performance_metrics, "expert_benchmarks", {})
        render_expert_benchmarks(expert_benchmarks)
    
    with performance_tabs[2]:
        # Render gating analysis view
        gating_evaluation = performance_metrics.get("gating_evaluation", {}) if isinstance(performance_metrics, dict) else getattr(performance_metrics, "gating_evaluation", {})
        render_gating_analysis(gating_evaluation)
    
    # Display metadata if available
    has_metadata = False
    metadata = {}
    
    if isinstance(performance_metrics, dict):
        has_metadata = "visualization_metadata" in performance_metrics
        metadata = performance_metrics.get("visualization_metadata", {})
    else:
        has_metadata = hasattr(performance_metrics, "visualization_metadata")
        if has_metadata:
            metadata = performance_metrics.visualization_metadata
    
    if has_metadata and metadata:
        with st.expander("Performance Evaluation Metadata", expanded=False):
            st.json(metadata)
    
    # Display experiment and data configuration IDs if available
    experiment_id = performance_metrics.get("experiment_id", "") if isinstance(performance_metrics, dict) else getattr(performance_metrics, "experiment_id", "")
    data_config_id = performance_metrics.get("data_config_id", "") if isinstance(performance_metrics, dict) else getattr(performance_metrics, "data_config_id", "")
    
    if experiment_id or data_config_id:
        st.sidebar.markdown("### Analysis Information")
        
        if experiment_id:
            st.sidebar.markdown(f"**Experiment ID:** `{experiment_id}`")
        
        if data_config_id:
            st.sidebar.markdown(f"**Data Config ID:** `{data_config_id}`")


if __name__ == "__main__":
    # This is for development/testing only
    import sys
    import os
    
    # Add the project root to the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    sys.path.insert(0, project_root)
    
    from moe_framework.interfaces.base import SystemState
    
    # Create a test system state
    test_state = SystemState()
    test_state.performance_metrics = {
        "expert_benchmarks": {},
        "gating_evaluation": {},
        "end_to_end_metrics": {},
        "visualization_metadata": {},
        "experiment_id": "test-experiment",
        "data_config_id": "test-config"
    }
    
    # Render the dashboard
    render_performance_analysis(test_state)
