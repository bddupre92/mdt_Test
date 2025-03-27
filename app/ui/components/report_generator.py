"""
Report Generator Component

This component provides a UI for generating interactive reports using the same
mechanism as the command-line tools.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import logging
import datetime
from pathlib import Path
import json
import importlib.util
from typing import Dict, Any, List, Optional, Tuple, Union

# Import the unified report generator
from app.reporting.unified_report_generator import get_report_generator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def render_report_generator():
    """
    Render the report generator component.
    """
    st.header("Interactive Report Generator")
    
    # Get the report generator
    report_generator = get_report_generator()
    
    # Create tabs for different report generation options
    tab1, tab2 = st.tabs(["Generate New Report", "View Existing Reports"])
    
    with tab1:
        st.subheader("Generate a New Interactive Report")
        
        # Select validation type
        validation_type = st.selectbox(
            "Select Validation Type",
            ["moe", "real_data"],
            format_func=lambda x: "MoE Validation" if x == "moe" else "Real Data Validation"
        )
        
        # Common parameters
        results_dir = st.text_input("Results Directory", value="results")
        
        # Report sections to include
        available_sections = report_generator.get_available_report_sections()
        
        # Create a section for report sections
        with st.expander("Report Sections to Include", expanded=True):
            st.info("Select which report sections to include in the generated report. If none are selected, all sections will be included.")
            
            # Create a multiselect for report sections
            selected_sections = st.multiselect(
                "Select Report Sections",
                options=[section["id"] for section in available_sections],
                default=[],
                format_func=lambda x: next((s["name"] for s in available_sections if s["id"] == x), x)
            )
        
        # Create a collapsible section for advanced parameters
        with st.expander("Advanced Parameters"):
            if validation_type == "moe":
                # MoE validation parameters
                components = st.multiselect(
                    "Components to Test",
                    ["all", "gating", "experts", "integration", "explainability"],
                    default=["all"]
                )
                
                benchmark_comparison = st.checkbox("Include Benchmark Comparison", value=False)
                explainers = st.multiselect(
                    "Explainers to Use",
                    ["all", "shap", "lime", "feature_importance"],
                    default=["all"]
                )
                
                # Drift notification parameters
                notify = st.checkbox("Enable Drift Notifications", value=False)
                notify_threshold = st.slider("Notification Threshold", 0.0, 1.0, 0.5, 0.1)
                notify_with_visuals = st.checkbox("Include Visuals in Notifications", value=False)
                
                # Retraining parameters
                enable_retraining = st.checkbox("Enable Selective Expert Retraining", value=False)
                retraining_threshold = st.slider("Retraining Threshold", 0.0, 1.0, 0.3, 0.1)
                
                # Continuous explainability parameters
                enable_continuous_explain = st.checkbox("Enable Continuous Explainability", value=False)
                continuous_explain_interval = st.number_input("Explanation Interval (seconds)", value=60)
                continuous_explain_types = st.multiselect(
                    "Explanation Types",
                    ["shap", "lime", "feature_importance"],
                    default=["shap", "feature_importance"]
                )
                
                # Confidence metrics parameters
                enable_confidence = st.checkbox("Enable Confidence Metrics", value=False)
                drift_weight = st.slider("Drift Weight", 0.0, 1.0, 0.5, 0.1)
                confidence_thresholds = st.multiselect(
                    "Confidence Thresholds",
                    [0.1, 0.3, 0.5, 0.7, 0.9],
                    default=[0.3, 0.5, 0.7, 0.9]
                )
                
                # Collect all parameters
                args_dict = {
                    'components': components,
                    'interactive': True,
                    'results_dir': results_dir,
                    'benchmark_comparison': benchmark_comparison,
                    'explainers': explainers,
                    'notify': notify,
                    'notify_threshold': notify_threshold,
                    'notify_with_visuals': notify_with_visuals,
                    'enable_retraining': enable_retraining,
                    'retraining_threshold': retraining_threshold,
                    'enable_continuous_explain': enable_continuous_explain,
                    'continuous_explain_interval': continuous_explain_interval,
                    'continuous_explain_types': continuous_explain_types,
                    'enable_confidence': enable_confidence,
                    'drift_weight': drift_weight,
                    'confidence_thresholds': confidence_thresholds
                }
            else:  # real_data validation
                # Real data validation parameters
                dataset = st.selectbox(
                    "Dataset",
                    ["migraine", "diabetes", "heart_disease", "custom"],
                    index=0
                )
                
                if dataset == "custom":
                    data_path = st.text_input("Custom Dataset Path")
                    args_dict = {
                        'dataset': dataset,
                        'data_path': data_path,
                        'results_dir': results_dir,
                        'interactive': True
                    }
                else:
                    args_dict = {
                        'dataset': dataset,
                        'results_dir': results_dir,
                        'interactive': True
                    }
        
        # Generate report button
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                try:
                    # Run validation and generate report with selected sections
                    result = report_generator.run_validation_and_generate_report(
                        validation_type, args_dict,
                        include_sections=selected_sections if selected_sections else None
                    )
                    
                    if result.get('success', False):
                        st.success(result.get('message', 'Report generated successfully'))
                        
                        # Display report path
                        if 'report_path' in result and result['report_path']:
                            report_path = result['report_path']
                            st.info(f"Report available at: {report_path}")
                            
                            # Display the report in an iframe
                            with open(report_path, 'r') as f:
                                report_html = f.read()
                            st.components.v1.html(report_html, height=600, scrolling=True)
                    else:
                        st.error(result.get('message', 'Failed to generate report'))
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    logger.error(f"Error generating report: {e}")
    
    with tab2:
        st.subheader("View Existing Reports")
        
        # Get all reports
        reports = report_generator.get_all_reports()
        
        if not reports:
            st.info("No reports found. Generate a new report first.")
        else:
            # Create two columns for report selection
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Display reports in a selectbox
                selected_report = st.selectbox(
                    "Select a Report",
                    reports,
                    format_func=lambda x: os.path.basename(x)
                )
            
            with col2:
                # Add a refresh button
                if st.button("🔄 Refresh Reports"):
                    st.experimental_rerun()
            
            if selected_report:
                # Display the report in an iframe
                with open(selected_report, 'r') as f:
                    report_html = f.read()
                st.components.v1.html(report_html, height=600, scrolling=True)

if __name__ == "__main__":
    render_report_generator()
