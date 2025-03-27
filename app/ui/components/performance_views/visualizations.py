"""
Performance Visualizations Component

This module provides components for advanced visualizations of MoE performance metrics,
including interactive plots and custom visualizations that aid in understanding model behavior.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

def render_visualizations(visualization_metadata: Dict[str, Any]):
    """
    Render advanced visualizations for MoE performance analysis.
    
    Args:
        visualization_metadata: Dictionary containing visualization metadata and configuration
    """
    st.header("Performance Visualizations")
    
    if not visualization_metadata:
        st.warning("No visualization metadata available")
        
        st.markdown("""
        ### Creating Performance Visualizations
        
        Visualizations have been integrated directly into each analysis component
        for better context and usability. Each component provides relevant 
        visualizations alongside the metrics and analysis results.
        
        To access visualizations:
        
        - Navigate to the specific analysis area (Expert Benchmarks, Gating Analysis, etc.)
        - Explore the visualizations provided in that context
        - Use the expandable sections to view detailed visualizations
        
        This integration provides better context and understanding of the data
        by showing visualizations right where they're most relevant.
        """)
        return
    
    # Inform users about the visualization integration
    st.info("Visualizations have been integrated into their respective analysis components.")
    st.markdown("""
    For improved organization and context, visualizations are now accessible
    directly within each analysis component:
    
    - **Expert Benchmarks**: Expert performance visualizations
    - **Gating Analysis**: Gating network behavior visualizations
    - **End-to-End Metrics**: Overall performance and comparison visualizations
    
    This integration provides better context by showing visualizations alongside
    their relevant metrics and analysis.
    """)
    
    # Show available visualization metadata if any exists
    if visualization_metadata:
        with st.expander("Visualization Configuration Metadata", expanded=False):
            st.json(visualization_metadata)
    
    # Add navigation buttons to the different visualization areas
    st.subheader("Quick Navigation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Expert Visualizations"):
            st.session_state.active_tab = "Expert Benchmarks"
            st.experimental_rerun()
    
    with col2:
        if st.button("Gating Visualizations"):
            st.session_state.active_tab = "Gating Network Analysis"
            st.experimental_rerun()
    
    with col3:
        if st.button("End-to-End Visualizations"):
            st.session_state.active_tab = "End-to-End Performance"
            st.experimental_rerun()
