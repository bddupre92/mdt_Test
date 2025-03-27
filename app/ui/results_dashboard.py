"""
Results Dashboard for MoE Framework

This module provides a Streamlit dashboard for the Results Management System,
integrating the existing interactive reports with new comparative analysis tools.
"""

import os
import sys
import streamlit as st
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the path to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the Results Management System
from app.ui.components.results_management import render_results_management_ui


def results_dashboard():
    """
    Main function for the Results Dashboard.
    """
    st.set_page_config(
        page_title="MoE Framework - Results Management",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add sidebar with navigation
    with st.sidebar:
        st.title("MoE Framework")
        st.subheader("Results Management")
        
        # Add navigation links
        st.markdown("### Navigation")
        st.markdown("- [Home](#)")
        st.markdown("- [Data Configuration](#)")
        st.markdown("- [Results Management](#)")
        
        # Add info about the dashboard
        st.markdown("---")
        st.markdown("""
        ### About
        This dashboard provides tools for managing, comparing, and exporting results 
        from the MoE framework. It integrates with the existing interactive reports 
        and adds comparative analysis capabilities.
        """)
    
    # Render the Results Management UI
    render_results_management_ui()


if __name__ == "__main__":
    results_dashboard()
