"""
Advanced Preprocessing Operations Component

This module provides Streamlit UI components for configuring advanced preprocessing operations
such as polynomial feature generation, dimensionality reduction, statistical feature generation,
and cluster-based feature generation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def render_advanced_operations():
    """Render UI for configuring advanced preprocessing operations."""
    st.subheader("Advanced Feature Engineering")
    
    # Get current configuration
    config = st.session_state.preprocessing_config
    advanced_ops = config.get('advanced_operations', {})
    
    # Polynomial Feature Generator
    st.write("---")
    st.write("**Polynomial Feature Generation**")
    
    poly_config = advanced_ops.get('polynomial_feature_generator', {})
    
    poly_include = st.checkbox(
        "Include Polynomial Features",
        value=poly_config.get('include', False),
        key="poly_include_checkbox"
    )
    
    if poly_include:
        poly_params = poly_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            degree = st.slider(
                "Polynomial Degree",
                min_value=2,
                max_value=5,
                value=int(poly_params.get('degree', 2)),
                step=1,
                key="poly_degree_slider"
            )
            poly_params['degree'] = degree
            
            include_bias = st.checkbox(
                "Include Bias Term",
                value=poly_params.get('include_bias', False),
                key="include_bias_checkbox"
            )
            poly_params['include_bias'] = include_bias
            
        with col2:
            interaction_only = st.checkbox(
                "Interaction Features Only",
                value=poly_params.get('interaction_only', True),
                key="interaction_only_checkbox"
            )
            poly_params['interaction_only'] = interaction_only
            
            exclude_cols = st.text_input(
                "Exclude Columns (comma-separated)",
                value=','.join(poly_params.get('exclude_cols', [])),
                key="poly_exclude_cols_input"
            )
            poly_params['exclude_cols'] = [col.strip() for col in exclude_cols.split(',')] if exclude_cols else []
            
        # Add help text explaining polynomial features
        with st.expander("About Polynomial Features"):
            st.markdown("""
            **Polynomial Features** transform your original features by creating new features that are polynomial combinations of the original ones.
            
            - **Degree**: Controls the highest power of the polynomial. Higher degrees capture more complex relationships but may lead to overfitting.
            - **Interaction Only**: When enabled, only creates interaction terms between different features (e.g., x1*x2) without powers of individual features.
            - **Include Bias**: Adds a constant term (1) to the feature set.
            
            Example with degree=2:
            - Original features: [a, b]
            - With interaction_only=True: [a, b, a*b]
            - With interaction_only=False: [a, b, a², b², a*b]
            """)
            
        poly_config['params'] = poly_params
        
    poly_config['include'] = poly_include
    advanced_ops['polynomial_feature_generator'] = poly_config
    
    # Dimensionality Reducer
    st.write("---")
    st.write("**Dimensionality Reduction**")
    
    dim_config = advanced_ops.get('dimensionality_reducer', {})
    
    dim_include = st.checkbox(
        "Include Dimensionality Reduction",
        value=dim_config.get('include', False),
        key="dim_include_checkbox"
    )
    
    if dim_include:
        dim_params = dim_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            dim_method = st.selectbox(
                "Reduction Method",
                options=['pca', 'kernel_pca', 'tsne'],
                index=['pca', 'kernel_pca', 'tsne'].index(
                    dim_params.get('method', 'pca')
                ),
                key="dim_method_select"
            )
            dim_params['method'] = dim_method
            
            n_components = st.number_input(
                "Number of Components",
                min_value=2,
                max_value=50,
                value=int(dim_params.get('n_components', 2)),
                step=1,
                key="n_components_input"
            )
            dim_params['n_components'] = n_components
            
        with col2:
            if dim_method == 'kernel_pca':
                kernel = st.selectbox(
                    "Kernel",
                    options=['rbf', 'poly', 'sigmoid', 'cosine'],
                    index=['rbf', 'poly', 'sigmoid', 'cosine'].index(
                        dim_params.get('kernel', 'rbf')
                    ),
                    key="kernel_select"
                )
                dim_params['kernel'] = kernel
                
            prefix = st.text_input(
                "Feature Name Prefix",
                value=dim_params.get('prefix', 'reduced'),
                key="prefix_input"
            )
            dim_params['prefix'] = prefix
            
            exclude_cols = st.text_input(
                "Exclude Columns (comma-separated)",
                value=','.join(dim_params.get('exclude_cols', [])),
                key="dim_exclude_cols_input"
            )
            dim_params['exclude_cols'] = [col.strip() for col in exclude_cols.split(',')] if exclude_cols else []
            
        # Add help text explaining dimensionality reduction
        with st.expander("About Dimensionality Reduction"):
            st.markdown("""
            **Dimensionality Reduction** techniques transform high-dimensional data into a lower-dimensional space while preserving important information.
            
            - **PCA (Principal Component Analysis)**: Linear technique that finds directions of maximum variance.
            - **Kernel PCA**: Non-linear extension of PCA that can capture more complex patterns.
            - **t-SNE**: Non-linear technique focused on preserving local structure, good for visualization.
            
            Benefits:
            - Reduces computational complexity
            - Helps with visualization
            - Can improve model performance by reducing noise
            - Addresses the "curse of dimensionality"
            """)
            
        dim_config['params'] = dim_params
        
    dim_config['include'] = dim_include
    advanced_ops['dimensionality_reducer'] = dim_config
    
    # Statistical Feature Generator
    st.write("---")
    st.write("**Statistical Feature Generation**")
    
    stat_config = advanced_ops.get('statistical_feature_generator', {})
    
    stat_include = st.checkbox(
        "Include Statistical Features",
        value=stat_config.get('include', False),
        key="stat_include_checkbox"
    )
    
    if stat_include:
        stat_params = stat_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            window_sizes_str = st.text_input(
                "Window Sizes (comma-separated)",
                value=','.join(str(w) for w in stat_params.get('window_sizes', [5, 10, 20])),
                key="window_sizes_input"
            )
            try:
                window_sizes = [int(w.strip()) for w in window_sizes_str.split(',') if w.strip()]
                stat_params['window_sizes'] = window_sizes if window_sizes else [5, 10, 20]
            except ValueError:
                st.error("Window sizes must be integers")
                stat_params['window_sizes'] = [5, 10, 20]
                
            group_by = st.text_input(
                "Group By Column (optional)",
                value=stat_params.get('group_by', ''),
                key="group_by_input"
            )
            stat_params['group_by'] = group_by if group_by else None
            
        with col2:
            available_stats = ['mean', 'std', 'min', 'max', 'skew', 'kurt']
            default_stats = stat_params.get('stats', ['mean', 'std'])
            
            selected_stats = st.multiselect(
                "Statistics to Calculate",
                options=available_stats,
                default=default_stats,
                key="stats_multiselect"
            )
            stat_params['stats'] = selected_stats if selected_stats else ['mean', 'std']
            
            exclude_cols = st.text_input(
                "Exclude Columns (comma-separated)",
                value=','.join(stat_params.get('exclude_cols', [])),
                key="stat_exclude_cols_input"
            )
            stat_params['exclude_cols'] = [col.strip() for col in exclude_cols.split(',')] if exclude_cols else []
            
        # Add help text explaining statistical features
        with st.expander("About Statistical Features"):
            st.markdown("""
            **Statistical Features** are derived from existing features by applying statistical operations over windows or groups.
            
            - **Window Sizes**: The number of consecutive samples to consider when calculating rolling statistics.
            - **Group By**: Calculate statistics grouped by a categorical variable (e.g., patient ID).
            - **Statistics**: Different statistical measures to calculate:
              - Mean: Average value
              - Std: Standard deviation (variability)
              - Min/Max: Minimum and maximum values
              - Skew: Asymmetry of the distribution
              - Kurt: "Tailedness" of the distribution
            
            These features can capture temporal patterns and relationships between variables.
            """)
            
        stat_config['params'] = stat_params
        
    stat_config['include'] = stat_include
    advanced_ops['statistical_feature_generator'] = stat_config
    
    # Cluster Feature Generator
    st.write("---")
    st.write("**Cluster-Based Feature Generation**")
    
    cluster_config = advanced_ops.get('cluster_feature_generator', {})
    
    cluster_include = st.checkbox(
        "Include Cluster Features",
        value=cluster_config.get('include', False),
        key="cluster_include_checkbox"
    )
    
    if cluster_include:
        cluster_params = cluster_config.get('params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=20,
                value=int(cluster_params.get('n_clusters', 3)),
                step=1,
                key="n_clusters_slider"
            )
            cluster_params['n_clusters'] = n_clusters
            
        with col2:
            cluster_method = st.selectbox(
                "Clustering Method",
                options=['kmeans'],  # Can be expanded in the future
                index=['kmeans'].index(
                    cluster_params.get('method', 'kmeans')
                ),
                key="cluster_method_select"
            )
            cluster_params['method'] = cluster_method
            
            exclude_cols = st.text_input(
                "Exclude Columns (comma-separated)",
                value=','.join(cluster_params.get('exclude_cols', [])),
                key="cluster_exclude_cols_input"
            )
            cluster_params['exclude_cols'] = [col.strip() for col in exclude_cols.split(',')] if exclude_cols else []
            
        # Add help text explaining cluster features
        with st.expander("About Cluster Features"):
            st.markdown("""
            **Cluster-Based Features** group similar data points together and create new features based on these clusters.
            
            - **Number of Clusters**: The number of groups to divide your data into.
            - **Clustering Method**: The algorithm used to identify clusters (currently K-means).
            
            New features created:
            - **cluster_id**: The cluster assignment for each data point
            - **distance_to_cluster_X**: The distance from each data point to each cluster center
            
            These features can help capture complex, non-linear relationships in the data and identify natural groupings.
            """)
            
        cluster_config['params'] = cluster_params
        
    cluster_config['include'] = cluster_include
    advanced_ops['cluster_feature_generator'] = cluster_config
    
    # Update the configuration
    config['advanced_operations'] = advanced_ops
    st.session_state.preprocessing_config = config
