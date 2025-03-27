"""
Baseline Comparisons Component

This module provides components for comparing MoE model performance
against baseline models, highlighting areas of improvement and weakness.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def render_baseline_comparisons(baseline_comparisons: Dict[str, Any]):
    """
    Render the baseline comparisons view for comparing MoE performance
    against baseline models.
    
    Args:
        baseline_comparisons: Dictionary containing baseline comparison data
    """
    st.header("Baseline Model Comparisons")
    
    if not baseline_comparisons:
        st.warning("No baseline comparison data available")
        
        st.markdown("""
        ### Setting Up Baseline Comparisons
        
        To generate baseline comparisons:
        
        1. Train baseline models using the same dataset as your MoE model
        2. Run the same evaluation metrics on all models
        3. Use the MoEMetricsCalculator to generate comparative metrics
        
        ```python
        from baseline_comparison.moe_metrics import MoEMetricsCalculator
        
        # Calculate baseline comparisons
        calculator = MoEMetricsCalculator()
        comparisons = calculator.compare_with_baselines(
            y_true=y_test,
            moe_predictions=moe_pred,
            baseline_predictions={
                "Linear Regression": lr_pred,
                "Random Forest": rf_pred,
                "XGBoost": xgb_pred
            }
        )
        
        # Update system state
        system_state.performance_metrics["baseline_comparisons"] = comparisons
        ```
        """)
        return
    
    # Redirect users to the integrated view in end_to_end_metrics
    st.info("Baseline comparisons have been integrated into the End-to-End Metrics view.")
    st.markdown("""
    For improved organization and analysis flow, baseline comparisons are now 
    accessible in the End-to-End Metrics view under the 'Baseline Comparison' tab.
    
    This integration provides better context by showing baseline comparisons alongside
    other end-to-end performance metrics.
    """)
    
    # Add a button to navigate to the end-to-end metrics view
    if st.button("Go to End-to-End Metrics"):
        st.session_state.active_tab = "End-to-End Performance"
        st.experimental_rerun()
