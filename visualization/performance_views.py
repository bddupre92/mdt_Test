"""
Performance Visualization Components

This module provides visualization components for analyzing MoE framework performance,
including expert contributions, gating network decisions, and overall metrics.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def visualize_expert_contributions(expert_data: Dict[str, Any]):
    """
    Visualize expert model contributions and performance.
    
    Args:
        expert_data: Dictionary containing expert performance data
    """
    st.subheader("Expert Model Contributions")
    
    # Expert contribution weights
    weights = expert_data.get('weights', {})
    if weights:
        fig = px.pie(
            values=list(weights.values()),
            names=list(weights.keys()),
            title="Expert Contribution Distribution"
        )
        st.plotly_chart(fig)
    
    # Expert performance metrics
    metrics = expert_data.get('metrics', {})
    if metrics:
        df = pd.DataFrame(metrics).T
        fig = px.bar(
            df,
            barmode='group',
            title="Expert Performance Metrics"
        )
        st.plotly_chart(fig)

def visualize_gating_decisions(gating_data: Dict[str, Any]):
    """
    Visualize gating network decisions and performance.
    
    Args:
        gating_data: Dictionary containing gating network data
    """
    st.subheader("Gating Network Analysis")
    
    # Decision confidence
    confidence = gating_data.get('confidence', [])
    if confidence:
        fig = px.histogram(
            confidence,
            title="Gating Decision Confidence Distribution",
            nbins=30
        )
        st.plotly_chart(fig)
    
    # Expert selection over time
    selections = gating_data.get('selections', [])
    if selections:
        df = pd.DataFrame(selections)
        fig = px.line(
            df,
            title="Expert Selection Over Time"
        )
        st.plotly_chart(fig)

def visualize_overall_performance(performance_data: Dict[str, Any]):
    """
    Visualize overall MoE system performance.
    
    Args:
        performance_data: Dictionary containing overall performance metrics
    """
    st.subheader("Overall System Performance")
    
    # Performance metrics
    metrics = performance_data.get('metrics', {})
    if metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "RMSE",
                f"{metrics.get('rmse', 0):.3f}",
                delta=metrics.get('rmse_change', 0)
            )
        
        with col2:
            st.metric(
                "MAE",
                f"{metrics.get('mae', 0):.3f}",
                delta=metrics.get('mae_change', 0)
            )
        
        with col3:
            st.metric(
                "R²",
                f"{metrics.get('r2', 0):.3f}",
                delta=metrics.get('r2_change', 0)
            )
    
    # Performance over time
    history = performance_data.get('history', [])
    if history:
        df = pd.DataFrame(history)
        fig = px.line(
            df,
            title="Performance History"
        )
        st.plotly_chart(fig)

def visualize_feature_importance(importance_data: Dict[str, float]):
    """
    Visualize feature importance across the MoE system.
    
    Args:
        importance_data: Dictionary mapping feature names to importance scores
    """
    st.subheader("Feature Importance Analysis")
    
    if importance_data:
        # Sort features by importance
        sorted_features = dict(sorted(
            importance_data.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        fig = px.bar(
            x=list(sorted_features.keys()),
            y=list(sorted_features.values()),
            title="Feature Importance Scores"
        )
        st.plotly_chart(fig)

def create_performance_dashboard(
    expert_data: Dict[str, Any],
    gating_data: Dict[str, Any],
    performance_data: Dict[str, Any],
    importance_data: Dict[str, float]
):
    """
    Create a complete performance dashboard with all visualizations.
    
    Args:
        expert_data: Expert performance data
        gating_data: Gating network data
        performance_data: Overall performance metrics
        importance_data: Feature importance scores
    """
    st.title("MoE Framework Performance Dashboard")
    
    # Overall performance metrics at the top
    visualize_overall_performance(performance_data)
    
    # Two-column layout for expert and gating visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        visualize_expert_contributions(expert_data)
    
    with col2:
        visualize_gating_decisions(gating_data)
    
    # Feature importance at the bottom
    visualize_feature_importance(importance_data) 