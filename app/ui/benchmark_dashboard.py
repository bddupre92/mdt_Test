"""
Benchmark Dashboard

This module provides the main dashboard for visualizing benchmark results and
accessing the MoE framework's Phase 2 components, including the Interactive Data
Configuration Dashboard and Results Management System.
"""

import streamlit as st

# Set page configuration first
st.set_page_config(
    page_title="MoE Framework Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import components after page config
from pathlib import Path
import os
import argparse
from typing import List, Dict, Any

# Import visualization components
from app.visualization.benchmark_visualizer import BenchmarkVisualizer

def main():
    """Main entry point for the dashboard."""
    st.title("MoE Framework Dashboard")
    
    # Get results directory from environment variable or command line
    # Parse command line arguments if provided
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    try:
        args, _ = parser.parse_known_args()
        results_dir = args.results_dir
    except:
        # Fall back to environment variable
        results_dir = os.environ.get("RESULTS_DIR", "results")
    
    # Initialize visualizer
    visualizer = BenchmarkVisualizer(results_dir=results_dir)
    
    # Get available result files
    results_path = Path(results_dir)
    if not results_path.exists():
        os.makedirs(results_path, exist_ok=True)
        result_files = []
    else:
        result_files = [
            f.name for f in results_path.glob("*.json")
            if f.is_file()
        ]
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Data Configuration", "Preprocessing Pipeline", "Results Management", "Comparison", "Meta Analysis", "Framework Runner", "Report Generator"]
    )
    
    # Import page components only when needed
    if page == "Overview":
        from app.ui.components.overview import render_overview
        render_overview(results_dir, result_files, visualizer)
    elif page == "Data Configuration":
        from app.ui.components.data_configuration import render_data_configuration_ui
        render_data_configuration_ui()
    elif page == "Preprocessing Pipeline":
        from app.ui.components.preprocessing_pipeline_component import render_preprocessing_pipeline
        render_preprocessing_pipeline()
    elif page == "Results Management":
        from app.ui.components.results_management import render_results_management_ui
        render_results_management_ui()
    elif page == "Comparison":
        from app.ui.components.comparison import render_comparison
        render_comparison(results_dir, result_files, visualizer)
    elif page == "Framework Runner":
        from app.ui.components.framework_runner import render_framework_runner
        render_framework_runner()
    elif page == "Report Generator":
        from app.ui.components.report_generator import render_report_generator
        render_report_generator()
    else:  # Meta Analysis
        from app.ui.components.meta_analysis import render_meta_analysis
        render_meta_analysis(results_dir, result_files, visualizer)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### MoE Framework - Phase 2
    
    This dashboard provides access to:
    - Interactive Data Configuration
    - Advanced Preprocessing Pipeline
    - Results Management System
    - Benchmark Analysis Tools
    - Interactive Report Generation
    """)
    
    from datetime import datetime
    st.sidebar.markdown(
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main() 