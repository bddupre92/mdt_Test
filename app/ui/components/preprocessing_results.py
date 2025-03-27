"""
Preprocessing Results Component

This module provides Streamlit UI components for visualizing and analyzing the results
of preprocessing operations, including data quality metrics, feature importance,
and transformed data previews.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


def render_preprocessing_results():
    """Render UI for visualizing preprocessing results."""
    st.subheader("Preprocessing Results")
    
    # Check if preprocessing results exist
    if 'preprocessing_results' not in st.session_state or st.session_state.preprocessing_results is None:
        st.info("Run preprocessing to see results here.")
        return
    
    # Get preprocessing results
    results = st.session_state.preprocessing_results
    
    # Create tabs for different result views
    result_tabs = st.tabs(["Overview", "Data Quality", "Feature Analysis", "Transformed Data"])
    
    # Overview tab
    with result_tabs[0]:
        render_overview_tab(results)
    
    # Data Quality tab
    with result_tabs[1]:
        render_data_quality_tab(results)
    
    # Feature Analysis tab
    with result_tabs[2]:
        render_feature_analysis_tab(results)
    
    # Transformed Data tab
    with result_tabs[3]:
        render_transformed_data_tab(results)


def render_overview_tab(results: Dict[str, Any]):
    """Render the overview tab with summary statistics and pipeline information.
    
    Args:
        results: Dictionary containing preprocessing results
    """
    st.write("### Preprocessing Overview")
    
    # Get transformed data
    transformed_data = results.get('transformed_data')
    original_data = results.get('original_data')
    
    if transformed_data is None or original_data is None:
        st.error("No data available for overview.")
        return
    
    # Display basic statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Rows Processed", 
            f"{len(original_data):,}", 
            f"{len(transformed_data) - len(original_data):,}",
            help="Number of rows in the original dataset, with delta showing rows added/removed"
        )
    
    with col2:
        original_features = len(original_data.columns)
        new_features = len(transformed_data.columns)
        st.metric(
            "Features", 
            f"{new_features:,}", 
            f"{new_features - original_features:,}",
            help="Number of features after preprocessing, with delta showing features added/removed"
        )
    
    with col3:
        # Calculate missing values percentage
        original_missing = original_data.isna().sum().sum() / (original_data.shape[0] * original_data.shape[1]) * 100
        transformed_missing = transformed_data.isna().sum().sum() / (transformed_data.shape[0] * transformed_data.shape[1]) * 100
        
        st.metric(
            "Missing Values", 
            f"{transformed_missing:.2f}%", 
            f"{transformed_missing - original_missing:.2f}%",
            help="Percentage of missing values after preprocessing, with delta showing change"
        )
    
    # Display pipeline operations summary
    st.write("### Pipeline Operations")
    
    pipeline_summary = results.get('pipeline_summary', {})
    operations = pipeline_summary.get('operations', [])
    
    if operations:
        # Create a table of operations
        operations_df = pd.DataFrame(operations)
        st.dataframe(operations_df)
    else:
        st.info("No operations were applied in this preprocessing run.")
    
    # Display execution time
    execution_time = results.get('execution_time', 0)
    st.write(f"**Total Execution Time:** {execution_time:.2f} seconds")
    
    # Display warnings if any
    warnings = results.get('warnings', [])
    if warnings:
        st.warning("**Preprocessing Warnings:**")
        for warning in warnings:
            st.write(f"- {warning}")


def render_data_quality_tab(results: Dict[str, Any]):
    """Render the data quality tab with quality metrics and visualizations.
    
    Args:
        results: Dictionary containing preprocessing results
    """
    st.write("### Data Quality Metrics")
    
    # Get data quality metrics
    quality_metrics = results.get('quality_metrics', {})
    
    if not quality_metrics:
        st.info("No data quality metrics available.")
        return
    
    # Display completeness metrics
    st.write("#### Completeness")
    
    completeness = quality_metrics.get('completeness', {})
    if completeness:
        # Create a bar chart of completeness by column
        completeness_df = pd.DataFrame({
            'Column': list(completeness.keys()),
            'Completeness (%)': [v * 100 for v in completeness.values()]
        })
        
        fig = px.bar(
            completeness_df,
            x='Column',
            y='Completeness (%)',
            title='Data Completeness by Column',
            labels={'Completeness (%)': 'Completeness (%)'},
            height=400
        )
        st.plotly_chart(fig)
    
    # Display consistency metrics
    st.write("#### Consistency")
    
    consistency = quality_metrics.get('consistency', {})
    if consistency:
        # Create a table of consistency metrics
        consistency_df = pd.DataFrame({
            'Metric': list(consistency.keys()),
            'Value': list(consistency.values())
        })
        st.dataframe(consistency_df)
    
    # Display outlier metrics
    st.write("#### Outliers")
    
    outliers = quality_metrics.get('outliers', {})
    if outliers:
        # Create a bar chart of outlier percentage by column
        outliers_df = pd.DataFrame({
            'Column': list(outliers.keys()),
            'Outlier (%)': [v * 100 for v in outliers.values()]
        })
        
        fig = px.bar(
            outliers_df,
            x='Column',
            y='Outlier (%)',
            title='Outlier Percentage by Column',
            labels={'Outlier (%)': 'Outlier (%)'},
            height=400
        )
        st.plotly_chart(fig)
    
    # Display distribution metrics
    st.write("#### Distribution Analysis")
    
    distribution = quality_metrics.get('distribution', {})
    if distribution:
        # Allow user to select a column to view distribution
        columns = list(distribution.keys())
        if columns:
            selected_column = st.selectbox("Select Column for Distribution Analysis", columns)
            
            if selected_column in distribution:
                dist_data = distribution[selected_column]
                
                # Display distribution metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean", f"{dist_data.get('mean', 0):.2f}")
                
                with col2:
                    st.metric("Median", f"{dist_data.get('median', 0):.2f}")
                
                with col3:
                    st.metric("Std Dev", f"{dist_data.get('std', 0):.2f}")
                
                with col4:
                    st.metric("Skewness", f"{dist_data.get('skew', 0):.2f}")
                
                # Display distribution plot if histogram data is available
                if 'histogram' in dist_data:
                    hist_data = dist_data['histogram']
                    
                    fig = px.histogram(
                        x=hist_data.get('values', []),
                        nbins=hist_data.get('bins', 20),
                        title=f'Distribution of {selected_column}',
                        labels={'x': selected_column, 'y': 'Frequency'},
                        height=400
                    )
                    st.plotly_chart(fig)


def render_feature_analysis_tab(results: Dict[str, Any]):
    """Render the feature analysis tab with feature importance and correlation analysis.
    
    Args:
        results: Dictionary containing preprocessing results
    """
    st.write("### Feature Analysis")
    
    # Get feature analysis results
    feature_analysis = results.get('feature_analysis', {})
    transformed_data = results.get('transformed_data')
    
    if not feature_analysis and transformed_data is None:
        st.info("No feature analysis available.")
        return
    
    # Feature importance
    st.write("#### Feature Importance")
    
    feature_importance = feature_analysis.get('feature_importance', {})
    if feature_importance:
        # Create a bar chart of feature importance
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=False)
        
        # Limit to top 20 features for readability
        if len(importance_df) > 20:
            importance_df = importance_df.head(20)
            st.info("Showing top 20 features by importance.")
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            title='Feature Importance',
            orientation='h',
            height=500
        )
        st.plotly_chart(fig)
    else:
        st.info("No feature importance information available.")
    
    # Correlation analysis
    st.write("#### Correlation Analysis")
    
    if transformed_data is not None:
        # Allow user to select correlation method
        correlation_method = st.radio(
            "Correlation Method",
            options=['pearson', 'spearman'],
            horizontal=True
        )
        
        # Get numeric columns for correlation
        numeric_cols = transformed_data.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            # Allow user to filter columns
            if len(numeric_cols) > 10:
                st.info(f"There are {len(numeric_cols)} numeric columns. You can select specific columns for correlation analysis.")
                selected_cols = st.multiselect(
                    "Select Columns for Correlation Analysis",
                    options=numeric_cols,
                    default=numeric_cols[:10]
                )
            else:
                selected_cols = numeric_cols
            
            if selected_cols:
                # Calculate correlation matrix
                corr_matrix = transformed_data[selected_cols].corr(method=correlation_method)
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    title=f'{correlation_method.capitalize()} Correlation Matrix',
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig)
                
                # Display top correlations
                st.write("**Top Correlations:**")
                
                # Get top absolute correlations
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_pairs.append((col1, col2, abs(corr_matrix.iloc[i, j])))
                
                # Sort by absolute correlation
                corr_pairs.sort(key=lambda x: x[2], reverse=True)
                
                # Display top 10 correlations
                top_corr = pd.DataFrame(
                    [(p[0], p[1], p[2]) for p in corr_pairs[:10]],
                    columns=['Feature 1', 'Feature 2', 'Absolute Correlation']
                )
                st.dataframe(top_corr)
        else:
            st.info("No numeric columns available for correlation analysis.")
    else:
        st.info("No transformed data available for correlation analysis.")
    
    # Feature clustering
    st.write("#### Feature Clustering")
    
    feature_clusters = feature_analysis.get('feature_clusters', {})
    if feature_clusters:
        # Display feature clusters
        for cluster_name, features in feature_clusters.items():
            st.write(f"**Cluster {cluster_name}:** {', '.join(features)}")
    else:
        st.info("No feature clustering information available.")


def render_transformed_data_tab(results: Dict[str, Any]):
    """Render the transformed data tab with preview of transformed dataset.
    
    Args:
        results: Dictionary containing preprocessing results
    """
    st.write("### Transformed Data Preview")
    
    # Get transformed data
    transformed_data = results.get('transformed_data')
    original_data = results.get('original_data')
    
    if transformed_data is None:
        st.error("No transformed data available.")
        return
    
    # Display data shape
    st.write(f"**Shape:** {transformed_data.shape[0]} rows × {transformed_data.shape[1]} columns")
    
    # Display data types summary
    dtypes_count = transformed_data.dtypes.value_counts().to_dict()
    dtypes_str = ", ".join([f"{count} {dtype}" for dtype, count in dtypes_count.items()])
    st.write(f"**Data Types:** {dtypes_str}")
    
    # Allow user to view original or transformed data
    data_view = st.radio(
        "Select Data View",
        options=["Transformed Data", "Original Data", "Side-by-Side Comparison"],
        horizontal=True
    )
    
    if data_view == "Transformed Data":
        # Allow user to filter columns
        if transformed_data.shape[1] > 10:
            st.info(f"There are {transformed_data.shape[1]} columns. You can select specific columns to view.")
            selected_cols = st.multiselect(
                "Select Columns to View",
                options=transformed_data.columns.tolist(),
                default=transformed_data.columns[:10].tolist()
            )
            
            if selected_cols:
                st.dataframe(transformed_data[selected_cols])
            else:
                st.dataframe(transformed_data)
        else:
            st.dataframe(transformed_data)
    
    elif data_view == "Original Data":
        if original_data is not None:
            # Allow user to filter columns
            if original_data.shape[1] > 10:
                st.info(f"There are {original_data.shape[1]} columns. You can select specific columns to view.")
                selected_cols = st.multiselect(
                    "Select Columns to View",
                    options=original_data.columns.tolist(),
                    default=original_data.columns[:10].tolist()
                )
                
                if selected_cols:
                    st.dataframe(original_data[selected_cols])
                else:
                    st.dataframe(original_data)
            else:
                st.dataframe(original_data)
        else:
            st.error("Original data not available.")
    
    else:  # Side-by-Side Comparison
        if original_data is not None:
            # Find common columns
            common_cols = list(set(original_data.columns) & set(transformed_data.columns))
            
            if common_cols:
                # Allow user to select a column to compare
                selected_col = st.selectbox(
                    "Select Column to Compare",
                    options=common_cols
                )
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    'Original': original_data[selected_col],
                    'Transformed': transformed_data[selected_col]
                })
                
                st.dataframe(comparison_df)
                
                # Create comparison visualization if numeric
                if pd.api.types.is_numeric_dtype(comparison_df['Original']) and pd.api.types.is_numeric_dtype(comparison_df['Transformed']):
                    st.write("**Distribution Comparison:**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=comparison_df['Original'],
                        name='Original',
                        opacity=0.7
                    ))
                    fig.add_trace(go.Histogram(
                        x=comparison_df['Transformed'],
                        name='Transformed',
                        opacity=0.7
                    ))
                    fig.update_layout(
                        title=f'Distribution Comparison for {selected_col}',
                        xaxis_title=selected_col,
                        yaxis_title='Count',
                        barmode='overlay'
                    )
                    st.plotly_chart(fig)
            else:
                st.warning("No common columns found between original and transformed data.")
        else:
            st.error("Original data not available for comparison.")
    
    # Download transformed data
    st.download_button(
        label="Download Transformed Data as CSV",
        data=transformed_data.to_csv(index=False).encode('utf-8'),
        file_name="transformed_data.csv",
        mime="text/csv"
    )
