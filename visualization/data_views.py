"""
Data Visualization Components

This module provides visualization components for analyzing data in the MoE framework,
including data distributions, correlations, and quality metrics.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def visualize_data_distribution(data: pd.DataFrame, feature: str):
    """
    Visualize the distribution of a single feature.
    
    Args:
        data: DataFrame containing the feature
        feature: Name of the feature to visualize
    """
    st.subheader(f"Distribution of {feature}")
    
    if pd.api.types.is_numeric_dtype(data[feature]):
        # Numeric feature
        fig = px.histogram(
            data,
            x=feature,
            nbins=30,
            title=f"Distribution of {feature}"
        )
        st.plotly_chart(fig)
        
        # Show basic statistics
        stats = data[feature].describe()
        st.write("Basic Statistics:")
        st.write(stats)
    else:
        # Categorical feature
        value_counts = data[feature].value_counts()
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Distribution of {feature}"
        )
        st.plotly_chart(fig)
        
        # Show frequency table
        st.write("Frequency Table:")
        st.write(value_counts)

def visualize_feature_correlations(data: pd.DataFrame, target: str):
    """Visualize correlations between features and target."""
    st.subheader("Feature Correlations")
    
    # Get only numeric columns
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    
    if numeric_data.empty:
        st.warning("No numeric features found for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        width=800,
        height=800
    )
    
    st.plotly_chart(fig)
    
    # Show correlation with target if target is numeric
    if target in numeric_data.columns:
        st.subheader(f"Correlations with {target}")
        
        # Sort correlations by absolute value
        target_corr = corr_matrix[target].sort_values(key=abs, ascending=False)
        
        # Create bar chart
        fig = go.Figure(data=go.Bar(
            x=target_corr.index,
            y=target_corr.values,
            marker_color=np.where(target_corr > 0, 'blue', 'red')
        ))
        
        fig.update_layout(
            title=f"Feature Correlations with {target}",
            xaxis_title="Features",
            yaxis_title="Correlation Coefficient",
            width=800,
            height=500
        )
        
        st.plotly_chart(fig)
    else:
        st.info(f"Target variable '{target}' is not numeric. Skipping target correlation analysis.")

def visualize_missing_values(data: pd.DataFrame):
    """
    Visualize missing values in the dataset.
    
    Args:
        data: DataFrame to analyze
    """
    st.subheader("Missing Values Analysis")
    
    # Calculate missing value percentages
    missing = (data.isnull().sum() / len(data)) * 100
    missing = missing[missing > 0].sort_values(ascending=True)
    
    if not missing.empty:
        fig = px.bar(
            x=missing.values,
            y=missing.index,
            orientation='h',
            title="Percentage of Missing Values by Feature"
        )
        st.plotly_chart(fig)
        
        # Show missing value statistics
        st.write("Missing Value Statistics:")
        st.write(pd.DataFrame({
            'Feature': missing.index,
            'Missing %': missing.values
        }))
    else:
        st.success("No missing values found in the dataset!")

def visualize_data_quality(data: pd.DataFrame):
    """
    Visualize various data quality metrics.
    
    Args:
        data: DataFrame to analyze
    """
    st.subheader("Data Quality Overview")
    
    # Basic dataset information
    st.write("Dataset Information:")
    st.write(f"Number of samples: {len(data)}")
    st.write(f"Number of features: {len(data.columns)}")
    
    # Data types
    st.write("\nFeature Data Types:")
    dtypes = pd.DataFrame(data.dtypes, columns=['Data Type'])
    st.write(dtypes)
    
    # Unique values count
    st.write("\nUnique Values Count:")
    unique_counts = pd.DataFrame(data.nunique(), columns=['Unique Values'])
    st.write(unique_counts)
    
    # Memory usage
    st.write("\nMemory Usage:")
    memory_usage = pd.DataFrame(data.memory_usage(deep=True) / 1024, columns=['Memory (KB)'])
    st.write(memory_usage)

def create_data_dashboard(
    data: pd.DataFrame,
    target: str = None,
    features_to_analyze: List[str] = None
):
    """
    Create a complete data analysis dashboard.
    
    Args:
        data: DataFrame to analyze
        target: Optional target variable
        features_to_analyze: Optional list of features to analyze in detail
    """
    st.title("MoE Framework Data Analysis Dashboard")
    
    # Data quality overview
    visualize_data_quality(data)
    
    # Missing values analysis
    visualize_missing_values(data)
    
    # Feature correlations
    if target:
        visualize_feature_correlations(data, target)
    
    # Individual feature distributions
    if features_to_analyze:
        for feature in features_to_analyze:
            if feature in data.columns:
                visualize_data_distribution(data, feature)
            else:
                st.warning(f"Feature '{feature}' not found in the dataset.") 