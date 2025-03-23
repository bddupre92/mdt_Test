"""
Benchmark Dashboard

This module provides the main dashboard for visualizing benchmark results.
"""

import streamlit as st

# Set page configuration first
st.set_page_config(
    page_title="Benchmark Dashboard",
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
    st.title("Benchmark Results Dashboard")
    
    # Get results directory from environment variable or command line
    # Parse command line arguments if provided
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="benchmark_results")
    try:
        args, _ = parser.parse_known_args()
        results_dir = args.results_dir
    except:
        # Fall back to environment variable
        results_dir = os.environ.get("RESULTS_DIR", "benchmark_results")
    
    # Initialize visualizer
    visualizer = BenchmarkVisualizer(results_dir=results_dir)
    
    # Get available result files
    results_path = Path(results_dir)
    if not results_path.exists():
        st.warning(f"No benchmark results found in {results_dir}. Run some benchmarks first.")
        st.info("You can still use the Framework Runner tab to run and visualize framework functions.")
        result_files = []
    else:
        result_files = [
            f.name for f in results_path.glob("*.json")
            if f.is_file()
        ]
        
        if not result_files:
            st.warning("No benchmark results found.")
            st.info("You can still use the Framework Runner tab to run and visualize framework functions.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Comparison", "Meta Analysis", "Framework Runner"]
    )
    
    # Import page components only when needed
    if page == "Overview":
        from app.ui.components.overview import render_overview
        render_overview(results_dir, result_files, visualizer)
    elif page == "Comparison":
        from app.ui.components.comparison import render_comparison
        render_comparison(results_dir, result_files, visualizer)
    elif page == "Framework Runner":
        from app.ui.components.framework_runner import render_framework_runner
        render_framework_runner()
    else:  # Meta Analysis
        from app.ui.components.meta_analysis import render_meta_analysis
        render_meta_analysis(results_dir, result_files, visualizer)
    
    # Footer
    st.sidebar.markdown("---")
    from datetime import datetime
    st.sidebar.markdown(
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main() 