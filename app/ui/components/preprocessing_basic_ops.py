"""
Basic Preprocessing Operations Component

This module provides Streamlit UI components for configuring basic preprocessing operations
such as missing value handling, outlier detection, feature scaling, and categorical encoding.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def render_basic_operations():
    """Render UI for configuring basic preprocessing operations."""
    st.subheader("Basic Preprocessing Operations")
    
    # Get current configuration
    config = st.session_state.preprocessing_config
    operations = config.get('operations', {})
    
    # Pipeline name
    pipeline_name = st.text_input(
        "Pipeline Name",
        value=config.get('pipeline_name', 'default_pipeline'),
        key="pipeline_name_input"
    )
    config['pipeline_name'] = pipeline_name
    
    # Data split configuration
    st.write("---")
    st.write("**Data Split Configuration**")
    
    split_config = config.get('data_split', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.5,
            value=split_config.get('test_size', 0.2),
            step=0.05,
            key="test_size_slider"
        )
        split_config['test_size'] = test_size
        
    with col2:
        val_size = st.slider(
            "Validation Set Size",
            min_value=0.1,
            max_value=0.5,
            value=split_config.get('validation_size', 0.25),
            step=0.05,
            key="val_size_slider"
        )
        split_config['validation_size'] = val_size
        
    with col3:
        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=1000,
            value=split_config.get('random_state', 42),
            step=1,
            key="random_state_input"
        )
        split_config['random_state'] = random_state
        
    stratify_column = st.text_input(
        "Stratify Column (leave empty to stratify by target)",
        value=split_config.get('stratify_column', ''),
        key="stratify_column_input"
    )
    split_config['stratify_column'] = stratify_column if stratify_column else None
    
    config['data_split'] = split_config
    
    # Missing Value Handler
    st.write("---")
    st.write("**Missing Value Handling**")
    
    missing_config = operations.get('missing_value_handler', {})
    
    missing_include = st.checkbox(
        "Include Missing Value Handler",
        value=missing_config.get('include', True),
        key="missing_include_checkbox"
    )
    
    if missing_include:
        missing_params = missing_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            missing_strategy = st.selectbox(
                "Numeric Strategy",
                options=['mean', 'median', 'most_frequent', 'constant'],
                index=['mean', 'median', 'most_frequent', 'constant'].index(
                    missing_params.get('strategy', 'mean')
                ),
                key="missing_strategy_select"
            )
            missing_params['strategy'] = missing_strategy
            
            if missing_strategy == 'constant':
                missing_fill_value = st.number_input(
                    "Fill Value",
                    value=missing_params.get('fill_value', 0),
                    key="missing_fill_value_input"
                )
                missing_params['fill_value'] = missing_fill_value
                
        with col2:
            cat_strategy = st.selectbox(
                "Categorical Strategy",
                options=['most_frequent', 'constant'],
                index=['most_frequent', 'constant'].index(
                    missing_params.get('categorical_strategy', 'most_frequent')
                ),
                key="cat_strategy_select"
            )
            missing_params['categorical_strategy'] = cat_strategy
            
        exclude_cols = st.text_input(
            "Exclude Columns (comma-separated)",
            value=','.join(missing_params.get('exclude_cols', [])),
            key="missing_exclude_cols_input"
        )
        missing_params['exclude_cols'] = [col.strip() for col in exclude_cols.split(',')] if exclude_cols else []
        
        missing_config['params'] = missing_params
        
    missing_config['include'] = missing_include
    operations['missing_value_handler'] = missing_config
    
    # Outlier Handler
    st.write("---")
    st.write("**Outlier Detection and Handling**")
    
    outlier_config = operations.get('outlier_handler', {})
    
    outlier_include = st.checkbox(
        "Include Outlier Handler",
        value=outlier_config.get('include', True),
        key="outlier_include_checkbox"
    )
    
    if outlier_include:
        outlier_params = outlier_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            outlier_method = st.selectbox(
                "Detection Method",
                options=['zscore', 'iqr'],
                index=['zscore', 'iqr'].index(
                    outlier_params.get('method', 'zscore')
                ),
                key="outlier_method_select"
            )
            outlier_params['method'] = outlier_method
            
            outlier_threshold = st.slider(
                "Threshold",
                min_value=1.0,
                max_value=5.0,
                value=float(outlier_params.get('threshold', 3.0)),
                step=0.1,
                key="outlier_threshold_slider"
            )
            outlier_params['threshold'] = outlier_threshold
            
        with col2:
            outlier_strategy = st.selectbox(
                "Handling Strategy",
                options=['winsorize', 'remove'],
                index=['winsorize', 'remove'].index(
                    outlier_params.get('strategy', 'winsorize')
                ),
                key="outlier_strategy_select"
            )
            outlier_params['strategy'] = outlier_strategy
            
        exclude_cols = st.text_input(
            "Exclude Columns (comma-separated)",
            value=','.join(outlier_params.get('exclude_cols', [])),
            key="outlier_exclude_cols_input"
        )
        outlier_params['exclude_cols'] = [col.strip() for col in exclude_cols.split(',')] if exclude_cols else []
        
        outlier_config['params'] = outlier_params
        
    outlier_config['include'] = outlier_include
    operations['outlier_handler'] = outlier_config
    
    # Feature Scaler
    st.write("---")
    st.write("**Feature Scaling**")
    
    scaler_config = operations.get('feature_scaler', {})
    
    scaler_include = st.checkbox(
        "Include Feature Scaler",
        value=scaler_config.get('include', True),
        key="scaler_include_checkbox"
    )
    
    if scaler_include:
        scaler_params = scaler_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            scaler_method = st.selectbox(
                "Scaling Method",
                options=['minmax', 'standard', 'robust'],
                index=['minmax', 'standard', 'robust'].index(
                    scaler_params.get('method', 'standard')
                ),
                key="scaler_method_select"
            )
            scaler_params['method'] = scaler_method
            
            if scaler_method == 'minmax':
                feature_range = st.slider(
                    "Feature Range",
                    min_value=0.0,
                    max_value=1.0,
                    value=(0.0, 1.0),
                    step=0.1,
                    key="feature_range_slider"
                )
                scaler_params['feature_range'] = feature_range
                
        with col2:
            exclude_cols = st.text_input(
                "Exclude Columns (comma-separated)",
                value=','.join(scaler_params.get('exclude_cols', [])),
                key="scaler_exclude_cols_input"
            )
            scaler_params['exclude_cols'] = [col.strip() for col in exclude_cols.split(',')] if exclude_cols else []
            
        scaler_config['params'] = scaler_params
        
    scaler_config['include'] = scaler_include
    operations['feature_scaler'] = scaler_config
    
    # Category Encoder
    st.write("---")
    st.write("**Categorical Encoding**")
    
    encoder_config = operations.get('category_encoder', {})
    
    encoder_include = st.checkbox(
        "Include Category Encoder",
        value=encoder_config.get('include', True),
        key="encoder_include_checkbox"
    )
    
    if encoder_include:
        encoder_params = encoder_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            encoder_method = st.selectbox(
                "Encoding Method",
                options=['label', 'onehot'],
                index=['label', 'onehot'].index(
                    encoder_params.get('method', 'onehot')
                ),
                key="encoder_method_select"
            )
            encoder_params['method'] = encoder_method
            
        with col2:
            exclude_cols = st.text_input(
                "Exclude Columns (comma-separated)",
                value=','.join(encoder_params.get('exclude_cols', [])),
                key="encoder_exclude_cols_input"
            )
            encoder_params['exclude_cols'] = [col.strip() for col in exclude_cols.split(',')] if exclude_cols else []
            
        encoder_config['params'] = encoder_params
        
    encoder_config['include'] = encoder_include
    operations['category_encoder'] = encoder_config
    
    # Feature Selector
    st.write("---")
    st.write("**Feature Selection**")
    
    selector_config = operations.get('feature_selector', {})
    
    selector_include = st.checkbox(
        "Include Feature Selector",
        value=selector_config.get('include', False),
        key="selector_include_checkbox"
    )
    
    if selector_include:
        selector_params = selector_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            selector_method = st.selectbox(
                "Selection Method",
                options=['variance', 'kbest', 'evolutionary'],
                index=['variance', 'kbest', 'evolutionary'].index(
                    selector_params.get('method', 'variance')
                ),
                key="selector_method_select"
            )
            selector_params['method'] = selector_method
            
            if selector_method == 'variance':
                threshold = st.slider(
                    "Variance Threshold",
                    min_value=0.0,
                    max_value=0.5,
                    value=float(selector_params.get('threshold', 0.01)),
                    step=0.01,
                    key="variance_threshold_slider"
                )
                selector_params['threshold'] = threshold
                
            elif selector_method == 'kbest':
                k = st.number_input(
                    "Number of Features (k)",
                    min_value=1,
                    max_value=100,
                    value=int(selector_params.get('k', 10)),
                    step=1,
                    key="kbest_k_input"
                )
                selector_params['k'] = k
                
        with col2:
            if selector_method == 'evolutionary':
                use_evolutionary = st.checkbox(
                    "Use Evolutionary Computation",
                    value=selector_params.get('use_evolutionary', True),
                    key="use_evolutionary_checkbox"
                )
                selector_params['use_evolutionary'] = use_evolutionary
                
                if use_evolutionary:
                    ec_algorithm = st.selectbox(
                        "Evolutionary Algorithm",
                        options=['aco', 'de', 'gwo'],
                        index=['aco', 'de', 'gwo'].index(
                            selector_params.get('ec_algorithm', 'aco')
                        ),
                        key="ec_algorithm_select"
                    )
                    selector_params['ec_algorithm'] = ec_algorithm
                    
            exclude_cols = st.text_input(
                "Exclude Columns (comma-separated)",
                value=','.join(selector_params.get('exclude_cols', [])),
                key="selector_exclude_cols_input"
            )
            selector_params['exclude_cols'] = [col.strip() for col in exclude_cols.split(',')] if exclude_cols else []
            
        selector_config['params'] = selector_params
        
    selector_config['include'] = selector_include
    operations['feature_selector'] = selector_config
    
    # Update the configuration
    config['operations'] = operations
    st.session_state.preprocessing_config = config
