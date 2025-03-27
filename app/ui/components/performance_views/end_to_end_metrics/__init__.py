"""
End-to-End Metrics View Components

This package contains modular components for end-to-end performance metrics visualization
and analysis in the MoE framework.
"""

from .overall_metrics import render_overall_metrics
from .temporal_analysis import render_temporal_analysis
from .baseline_comparison import render_baseline_comparison
from .statistical_tests import render_statistical_tests

__all__ = [
    'render_overall_metrics',
    'render_temporal_analysis',
    'render_baseline_comparison',
    'render_statistical_tests',
    'render_end_to_end_metrics'
]

def render_end_to_end_metrics(end_to_end_metrics):
    """
    Render the End-to-End Performance Metrics view with all components.
    
    Args:
        end_to_end_metrics: Dictionary containing end-to-end performance metrics
    """
    import streamlit as st
    
    st.header("End-to-End Performance Metrics")
    
    st.markdown("""
    This view provides comprehensive end-to-end performance metrics for your MoE system,
    including overall performance, temporal analysis, and comparison to baseline models.
    """)
    
    if not end_to_end_metrics:
        st.warning("No end-to-end performance metrics available")
        
        st.markdown("""
        ### Getting Started with End-to-End Metrics
        
        To generate end-to-end performance metrics:
        
        1. Run the MoE system on a test dataset
        2. Calculate comprehensive metrics using the MoEMetricsCalculator
        3. Save the metrics to the system state
        
        Example code:
        ```python
        from baseline_comparison.moe_metrics import MoEMetricsCalculator
        
        calculator = MoEMetricsCalculator()
        metrics = calculator.compute_all_metrics(
            y_true=y_test,
            y_pred=predictions,
            expert_weights=expert_weights,
            expert_outputs=individual_expert_outputs
        )
        
        # Update the system state
        system_state.performance_metrics["end_to_end_metrics"] = metrics
        state_manager.save_state(system_state, "path/to/checkpoint")
        ```
        """)
        return
    
    # Create tabs for different metrics views
    metric_tabs = st.tabs([
        "Overall Metrics", 
        "Temporal Analysis", 
        "Baseline Comparison",
        "Statistical Tests"
    ])
    
    with metric_tabs[0]:
        render_overall_metrics(end_to_end_metrics)
    
    with metric_tabs[1]:
        render_temporal_analysis(end_to_end_metrics)
    
    with metric_tabs[2]:
        render_baseline_comparison(end_to_end_metrics)
    
    with metric_tabs[3]:
        render_statistical_tests(end_to_end_metrics)
