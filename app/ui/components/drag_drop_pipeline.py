"""
Drag and Drop Pipeline Builder Component

This module provides a drag-and-drop interface for building preprocessing pipelines.
It uses Streamlit's components and session state to create an interactive experience.
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid

# Import preprocessing pipeline
from data.preprocessing_pipeline import (
    PreprocessingPipeline, 
    MissingValueHandler, 
    OutlierHandler, 
    FeatureScaler, 
    CategoryEncoder, 
    FeatureSelector,
    TimeSeriesProcessor
)

# Operation type definitions with icons and descriptions
OPERATION_TYPES = {
    "MissingValueHandler": {
        "icon": "❓",
        "description": "Handle missing values in the dataset",
        "color": "#FF9F1C"
    },
    "OutlierHandler": {
        "icon": "⚠️",
        "description": "Detect and handle outliers",
        "color": "#E71D36"
    },
    "FeatureScaler": {
        "icon": "📏",
        "description": "Scale features to a standard range",
        "color": "#2EC4B6"
    },
    "CategoryEncoder": {
        "icon": "🏷️",
        "description": "Encode categorical variables",
        "color": "#011627"
    },
    "FeatureSelector": {
        "icon": "🔍",
        "description": "Select important features",
        "color": "#6A0572"
    },
    "TimeSeriesProcessor": {
        "icon": "⏱️",
        "description": "Process time series data",
        "color": "#0B6E4F"
    }
}

def render_drag_drop_pipeline_builder():
    """Render the drag-and-drop visual pipeline builder component."""
    st.subheader("Visual Pipeline Builder")
    
    # Initialize session state for pipeline if not exists
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = PreprocessingPipeline(name="Custom Pipeline")
    
    if 'pipeline_operations' not in st.session_state:
        st.session_state.pipeline_operations = []
    
    if 'dragging' not in st.session_state:
        st.session_state.dragging = None
    
    if 'drop_target' not in st.session_state:
        st.session_state.drop_target = None
    
    # File uploader for data
    uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            st.write(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            
            # Display data sample
            with st.expander("Data Sample", expanded=False):
                st.dataframe(data.head())
            
            # Pipeline builder
            col1, col2 = st.columns([1, 3])
            
            # Operations palette
            with col1:
                st.subheader("Operations")
                for op_type, op_info in OPERATION_TYPES.items():
                    operation_card(op_type, op_info)
            
            # Pipeline canvas
            with col2:
                st.subheader("Pipeline Canvas")
                pipeline_canvas()
            
            # Execute pipeline button
            if len(st.session_state.pipeline_operations) > 0:
                if st.button("Execute Pipeline", key="execute_pipeline"):
                    with st.spinner("Executing pipeline..."):
                        execute_pipeline(data)
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        st.info("Please upload a data file to begin.")

def operation_card(op_type, op_info):
    """Render a draggable operation card."""
    card_id = f"op_{op_type}_{uuid.uuid4().hex[:8]}"
    
    # Create a card-like container
    card = st.container()
    with card:
        # Use columns for layout
        icon_col, text_col = st.columns([1, 4])
        with icon_col:
            st.markdown(f"<div style='font-size:24px;'>{op_info['icon']}</div>", unsafe_allow_html=True)
        with text_col:
            st.markdown(f"**{op_type}**")
            st.markdown(f"<small>{op_info['description']}</small>", unsafe_allow_html=True)
        
        # Add a "drag" button
        if st.button("Add to Pipeline", key=card_id):
            add_operation_to_pipeline(op_type)

def add_operation_to_pipeline(op_type):
    """Add an operation to the pipeline."""
    # Create a unique ID for this operation
    op_id = f"{op_type}_{uuid.uuid4().hex[:8]}"
    
    # Configure the operation
    if op_type == "MissingValueHandler":
        operation = configure_missing_value_handler()
    elif op_type == "OutlierHandler":
        operation = configure_outlier_handler()
    elif op_type == "FeatureScaler":
        operation = configure_feature_scaler()
    elif op_type == "CategoryEncoder":
        operation = configure_category_encoder()
    elif op_type == "FeatureSelector":
        operation = configure_feature_selector()
    elif op_type == "TimeSeriesProcessor":
        operation = configure_time_series_processor()
    
    # Add to pipeline operations
    if operation:
        st.session_state.pipeline_operations.append({
            "id": op_id,
            "type": op_type,
            "operation": operation,
            "description": f"{op_type}: {operation.get_description()}"
        })
        
        # Add to pipeline
        st.session_state.pipeline.add_operation(operation)
        
        # Rerun to update the UI
        st.experimental_rerun()

def pipeline_canvas():
    """Render the pipeline canvas where operations can be arranged."""
    if not st.session_state.pipeline_operations:
        st.info("Drag operations from the palette to build your pipeline.")
        return
    
    # Display current pipeline operations
    for i, op in enumerate(st.session_state.pipeline_operations):
        op_type = op['type']
        op_info = OPERATION_TYPES.get(op_type, {
            "icon": "🔧",
            "color": "#888888"
        })
        
        # Create a card for the operation
        col1, col2, col3, col4 = st.columns([1, 6, 1, 1])
        
        with col1:
            st.markdown(f"<div style='font-size:24px;'>{op_info['icon']}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**{i+1}. {op_type}**")
            st.markdown(f"<small>{op['description']}</small>", unsafe_allow_html=True)
        
        with col3:
            if st.button("⬆️", key=f"up_{i}", disabled=i==0):
                # Move operation up
                st.session_state.pipeline_operations.insert(i-1, st.session_state.pipeline_operations.pop(i))
                st.session_state.pipeline.reorder_operations(i, i-1)
                st.experimental_rerun()
            
            if st.button("⬇️", key=f"down_{i}", disabled=i==len(st.session_state.pipeline_operations)-1):
                # Move operation down
                st.session_state.pipeline_operations.insert(i+1, st.session_state.pipeline_operations.pop(i))
                st.session_state.pipeline.reorder_operations(i, i+1)
                st.experimental_rerun()
        
        with col4:
            if st.button("✏️", key=f"edit_{i}"):
                # Edit operation
                st.session_state.edit_operation_index = i
                st.session_state.edit_operation = True
                st.experimental_rerun()
            
            if st.button("🗑️", key=f"delete_{i}"):
                # Remove operation
                st.session_state.pipeline_operations.pop(i)
                st.session_state.pipeline.remove_operation(i)
                st.experimental_rerun()
        
        # Add a separator
        if i < len(st.session_state.pipeline_operations) - 1:
            st.markdown("<div style='text-align:center; margin:10px 0;'>⬇️</div>", unsafe_allow_html=True)

def configure_missing_value_handler():
    """Configure a missing value handler operation."""
    with st.expander("Configure Missing Value Handler", expanded=True):
        strategy = st.selectbox(
            "Strategy",
            ["mean", "median", "most_frequent", "constant", "knn"],
            index=0
        )
        
        fill_value = None
        if strategy == "constant":
            fill_value = st.number_input("Fill Value", value=0)
        
        n_neighbors = 5
        if strategy == "knn":
            n_neighbors = st.number_input("Number of Neighbors", min_value=1, max_value=20, value=5)
        
        # Create the operation
        operation = MissingValueHandler(
            strategy=strategy,
            fill_value=fill_value,
            n_neighbors=n_neighbors if strategy == "knn" else None
        )
        
        if st.button("Add to Pipeline", key="add_missing_handler"):
            return operation
    
    return None

def configure_outlier_handler():
    """Configure an outlier handler operation."""
    with st.expander("Configure Outlier Handler", expanded=True):
        method = st.selectbox(
            "Method",
            ["iqr", "z_score", "isolation_forest", "dbscan"],
            index=0
        )
        
        threshold = 1.5
        if method in ["iqr", "z_score"]:
            threshold = st.number_input(
                "Threshold" if method == "z_score" else "IQR Factor",
                min_value=0.1,
                max_value=5.0,
                value=1.5 if method == "iqr" else 3.0,
                step=0.1
            )
        
        strategy = st.selectbox(
            "Strategy",
            ["clip", "remove", "mean", "median"],
            index=0
        )
        
        # Create the operation
        operation = OutlierHandler(
            method=method,
            threshold=threshold if method in ["iqr", "z_score"] else None,
            strategy=strategy
        )
        
        if st.button("Add to Pipeline", key="add_outlier_handler"):
            return operation
    
    return None

def configure_feature_scaler():
    """Configure a feature scaler operation."""
    with st.expander("Configure Feature Scaler", expanded=True):
        scaler_type = st.selectbox(
            "Scaler Type",
            ["standard", "minmax", "robust", "normalizer"],
            index=0
        )
        
        # Additional parameters based on scaler type
        feature_range = (0, 1)
        if scaler_type == "minmax":
            min_val = st.number_input("Min Value", value=0.0)
            max_val = st.number_input("Max Value", value=1.0)
            feature_range = (min_val, max_val)
        
        # Create the operation
        operation = FeatureScaler(
            scaler_type=scaler_type,
            feature_range=feature_range if scaler_type == "minmax" else None
        )
        
        if st.button("Add to Pipeline", key="add_feature_scaler"):
            return operation
    
    return None

def configure_category_encoder():
    """Configure a category encoder operation."""
    with st.expander("Configure Category Encoder", expanded=True):
        encoding_method = st.selectbox(
            "Encoding Method",
            ["one_hot", "label", "ordinal", "target", "frequency"],
            index=0
        )
        
        handle_unknown = st.selectbox(
            "Handle Unknown",
            ["error", "ignore", "infrequent_if_exist"],
            index=1
        )
        
        # Create the operation
        operation = CategoryEncoder(
            encoding_method=encoding_method,
            handle_unknown=handle_unknown
        )
        
        if st.button("Add to Pipeline", key="add_category_encoder"):
            return operation
    
    return None

def configure_feature_selector():
    """Configure a feature selector operation."""
    with st.expander("Configure Feature Selector", expanded=True):
        method = st.selectbox(
            "Selection Method",
            ["variance", "k_best", "rfe", "from_model", "genetic"],
            index=0
        )
        
        # Parameters based on method
        threshold = 0.0
        k = 10
        
        if method == "variance":
            threshold = st.number_input("Variance Threshold", min_value=0.0, value=0.0, step=0.01)
        elif method in ["k_best", "rfe"]:
            k = st.number_input("Number of Features to Select", min_value=1, value=10)
        
        # Create the operation
        operation = FeatureSelector(
            method=method,
            threshold=threshold if method == "variance" else None,
            k=k if method in ["k_best", "rfe"] else None
        )
        
        if st.button("Add to Pipeline", key="add_feature_selector"):
            return operation
    
    return None

def configure_time_series_processor():
    """Configure a time series processor operation."""
    with st.expander("Configure Time Series Processor", expanded=True):
        operations = st.multiselect(
            "Operations",
            ["lag_features", "rolling_statistics", "ewm", "diff", "date_features"],
            default=["lag_features"]
        )
        
        # Parameters based on selected operations
        lag_periods = [1]
        window_size = 3
        
        if "lag_features" in operations:
            lag_input = st.text_input("Lag Periods (comma-separated)", "1,2,3")
            try:
                lag_periods = [int(x.strip()) for x in lag_input.split(",")]
            except:
                st.warning("Invalid lag periods. Using default [1].")
                lag_periods = [1]
        
        if "rolling_statistics" in operations or "ewm" in operations:
            window_size = st.number_input("Window Size", min_value=2, value=3)
        
        # Create the operation
        operation = TimeSeriesProcessor(
            operations=operations,
            lag_periods=lag_periods if "lag_features" in operations else None,
            window_size=window_size if "rolling_statistics" in operations or "ewm" in operations else None
        )
        
        if st.button("Add to Pipeline", key="add_time_series_processor"):
            return operation
    
    return None

def execute_pipeline(data):
    """Execute the preprocessing pipeline on the data."""
    if 'pipeline' not in st.session_state or len(st.session_state.pipeline.operations) == 0:
        st.warning("No pipeline operations defined.")
        return
    
    try:
        # Execute the pipeline
        processed_data = st.session_state.pipeline.transform(data)
        
        # Store the processed data
        st.session_state.processed_data = processed_data
        
        # Display results
        st.success("Pipeline executed successfully!")
        
        # Show processed data
        st.subheader("Processed Data")
        st.dataframe(processed_data.head())
        
        # Show data shape comparison
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Rows", data.shape[0])
            st.metric("Original Columns", data.shape[1])
        
        with col2:
            st.metric("Processed Rows", processed_data.shape[0], 
                      delta=processed_data.shape[0] - data.shape[0])
            st.metric("Processed Columns", processed_data.shape[1],
                      delta=processed_data.shape[1] - data.shape[1])
        
        # Option to download processed data
        csv = processed_data.to_csv(index=False)
        st.download_button(
            label="Download Processed Data",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error executing pipeline: {str(e)}")
