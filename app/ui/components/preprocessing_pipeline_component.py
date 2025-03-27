"""
Preprocessing Pipeline Component

This module provides a Streamlit component for configuring and managing the preprocessing pipeline.
It integrates with the preprocessing manager to provide a user-friendly interface for
configuring, optimizing, and applying preprocessing operations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import plotly.express as px
import plotly.graph_objects as go
import time

from data.preprocessing_manager import PreprocessingManager
from app.ui.components.preprocessing_basic_ops import render_basic_operations
from app.ui.components.preprocessing_advanced_ops import render_advanced_operations
from app.ui.components.preprocessing_domain_ops import render_domain_operations
from app.ui.components.preprocessing_optimization import render_optimization_section
from app.ui.components.preprocessing_results import render_preprocessing_results


def render_preprocessing_pipeline(data: Optional[pd.DataFrame] = None):
    """Render the preprocessing pipeline component.
    
    Args:
        data: Optional DataFrame to use for preprocessing
    """
    st.header("Automated Preprocessing Pipeline")
    
    # Initialize session state for preprocessing
    if 'preprocessing_manager' not in st.session_state:
        st.session_state.preprocessing_manager = PreprocessingManager()
        
    if 'preprocessing_config' not in st.session_state:
        st.session_state.preprocessing_config = st.session_state.preprocessing_manager.get_config()
        
    if 'preprocessing_results' not in st.session_state:
        st.session_state.preprocessing_results = None
        
    if 'active_preprocessing_tab' not in st.session_state:
        st.session_state.active_preprocessing_tab = "Basic Operations"
        
    # Allow data upload when called directly from the dashboard
    if data is None:
        st.info("No data loaded. Please upload a dataset to use the preprocessing pipeline.")
        uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.session_state.data = data
                st.success(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
                
                # Display data sample
                st.subheader("Data Sample")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        else:
            # Use sample data for demonstration if no data is uploaded
            if st.button("Use Sample Data"):
                # Generate sample data
                np.random.seed(42)
                sample_data = pd.DataFrame({
                    'age': np.random.normal(45, 15, 100).astype(int),
                    'gender': np.random.choice(['M', 'F'], 100),
                    'pain_level': np.random.randint(1, 11, 100),
                    'duration_hours': np.random.exponential(4, 100),
                    'medication_response': np.random.choice(['none', 'mild', 'moderate', 'significant'], 100),
                    'comorbidity_count': np.random.poisson(1.5, 100)
                })
                
                # Add some missing values
                for col in sample_data.columns:
                    mask = np.random.random(len(sample_data)) < 0.1
                    sample_data.loc[mask, col] = np.nan
                
                data = sample_data
                st.session_state.data = data
                st.success("Sample data loaded for demonstration")
                
                # Display data sample
                st.subheader("Data Sample")
                st.dataframe(data.head())
        
    # Function to update configuration
    def update_config():
        st.session_state.preprocessing_manager.update_config(st.session_state.preprocessing_config)
        
    # Function to save configuration
    def save_config():
        config_dir = os.path.join("configs", "preprocessing")
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = os.path.join(config_dir, f"{st.session_state.preprocessing_config.get('pipeline_name', 'default')}.json")
        st.session_state.preprocessing_manager.save_config(config_path)
        st.success(f"Configuration saved to {config_path}")
        
    # Function to load configuration
    def load_config(config_path):
        st.session_state.preprocessing_manager.load_config(config_path)
        st.session_state.preprocessing_config = st.session_state.preprocessing_manager.get_config()
        st.success(f"Configuration loaded from {config_path}")
        
    # Function to run preprocessing
    def run_preprocessing():
        if data is None:
            st.error("No data available for preprocessing")
            return
            
        with st.spinner("Running preprocessing..."):
            # Update configuration before running
            update_config()
            
            # Get target column if specified
            target_col = st.session_state.preprocessing_config.get('optimization', {}).get('params', {}).get('target_col')
            
            # Run preprocessing
            results = st.session_state.preprocessing_manager.preprocess_data(data, target_col)
            st.session_state.preprocessing_results = results
            
            st.success("Preprocessing complete!")
            
    # Function to run optimization
    def run_optimization():
        if data is None:
            st.error("No data available for optimization")
            return
            
        with st.spinner("Optimizing preprocessing pipeline..."):
            # Update configuration before running
            update_config()
            
            # Run optimization
            optimized_pipeline = st.session_state.preprocessing_manager.optimize_pipeline(data)
            
            # Update configuration with optimized parameters
            st.session_state.preprocessing_config = st.session_state.preprocessing_manager.get_config()
            
            st.success("Pipeline optimization complete!")
        
    # Create tabs for different sections
    tabs = ["Basic Operations", "Advanced Operations", "Domain-Specific", "Optimization", "Results"]
    
    # Use radio buttons for tabs to save vertical space
    st.session_state.active_preprocessing_tab = st.radio("Configuration Sections:", tabs, 
                                                        index=tabs.index(st.session_state.active_preprocessing_tab))
    
    st.write("---")
    
    # Render the active tab
    if st.session_state.active_preprocessing_tab == "Basic Operations":
        render_basic_operations()
        
    elif st.session_state.active_preprocessing_tab == "Advanced Operations":
        render_advanced_operations()
        
    elif st.session_state.active_preprocessing_tab == "Domain-Specific":
        render_domain_operations()
        
    elif st.session_state.active_preprocessing_tab == "Optimization":
        render_optimization_section()
        
    elif st.session_state.active_preprocessing_tab == "Results":
        render_preprocessing_results()
    
    # Add buttons for actions
    st.write("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Run Preprocessing"):
            run_preprocessing()
            
    with col2:
        if st.button("Optimize Pipeline"):
            run_optimization()
            
    with col3:
        if st.button("Save Configuration"):
            save_config()
            
    with col4:
        # List available configurations
        config_dir = os.path.join("configs", "preprocessing")
        os.makedirs(config_dir, exist_ok=True)
        
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
        
        if config_files:
            selected_config = st.selectbox("Load Configuration", config_files)
            
            if st.button("Load"):
                config_path = os.path.join(config_dir, selected_config)
                load_config(config_path)
        else:
            st.write("No saved configurations")
            
    # Display pipeline summary
    st.write("---")
    st.subheader("Pipeline Summary")
    
    pipeline_summary = st.session_state.preprocessing_manager.get_pipeline_summary()
    
    st.write(f"Pipeline Name: {pipeline_summary['name']}")
    st.write(f"Number of Operations: {pipeline_summary['operations_count']}")
    
    if pipeline_summary['operations_count'] > 0:
        # Create a table of operations
        operations_df = pd.DataFrame(pipeline_summary['operations'])
        st.dataframe(operations_df)
