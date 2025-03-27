"""
Comparison Component

This module provides the comparison page component for the benchmark dashboard.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def render_comparison(results_dir: str, result_files: List[str], visualizer):
    """Render the comparison page.
    
    Args:
        results_dir: Directory containing benchmark results
        result_files: List of available result files
        visualizer: Instance of BenchmarkVisualizer
    """
    import streamlit as st  # Import inside function
    import os
    from pathlib import Path
    
    # Create tabs for different comparison types
    comparison_tabs = st.tabs(["Optimizer Comparison", "Interactive Reports Comparison"])
    
    # Tab 1: Optimizer Comparison (original functionality)
    with comparison_tabs[0]:
        st.header("Optimizer Comparison")
        
        if not result_files:
            st.warning("No benchmark results found.")
        else:
            # Filter for comparison results
            comparison_files = [f for f in result_files if is_comparison_result(f, visualizer)]
            
            if not comparison_files:
                st.warning("No comparison results found. Run a comparison benchmark first.")
            else:
                # Select comparison result to view
                selected_file = st.selectbox(
                    "Select a comparison result to view",
                    comparison_files
                )
                
                if selected_file:
                    display_comparison_details(selected_file, visualizer)
    
    # Tab 2: Interactive Reports Comparison
    with comparison_tabs[1]:
        st.header("Interactive Reports Comparison")
        
        # Find all interactive reports
        interactive_reports = find_interactive_reports()
        
        if not interactive_reports:
            st.warning("No interactive reports found.")
        else:
            st.info(f"Found {len(interactive_reports)} interactive reports for comparison.")
            
            # Allow multi-selection of reports to compare
            selected_reports = st.multiselect(
                "Select reports to compare",
                interactive_reports,
                format_func=lambda x: Path(x).name
            )
            
            if selected_reports:
                display_interactive_reports(selected_reports)

def is_comparison_result(filename: str, visualizer) -> bool:
    """Check if a result file contains comparison data.
    
    Args:
        filename: Name of the result file
        visualizer: Instance of BenchmarkVisualizer
        
    Returns:
        True if file contains comparison data, False otherwise
    """
    try:
        result = visualizer.load_results(filename)
        return "optimizers" in result and isinstance(result["optimizers"], dict)
    except:
        return False

def display_comparison_details(selected_file: str, visualizer):
    """Display detailed comparison information.
    
    Args:
        selected_file: Name of the selected result file
        visualizer: Instance of BenchmarkVisualizer
    """
    import streamlit as st  # Import inside function
    
    try:
        result = visualizer.load_results(selected_file)
        
        # Display basic information
        st.subheader("Comparison Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Benchmark Function:** {result.get('benchmark_name', 'Unknown')}")
            st.write(f"**Dimension:** {result.get('dimension', 'Unknown')}")
            st.write(f"**Bounds:** {result.get('bounds', 'Unknown')}")
        
        with col2:
            st.write(f"**Max Evaluations:** {result.get('max_evaluations', 'Unknown')}")
            st.write(f"**Number of Runs:** {result.get('num_runs', 'Unknown')}")
            st.write(f"**Timestamp:** {result.get('timestamp', 'Unknown')}")
        
        # Display comparison table
        st.subheader("Performance Comparison")
        comparison_data = create_comparison_table(result)
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
        
        # Display convergence plot
        st.subheader("Convergence Comparison")
        fig = visualizer.plot_comparison(
            result,
            title=f"Convergence Comparison for {result.get('benchmark_name', 'Unknown')}"
        )
        if fig:
            st.pyplot(fig)
            plt.close(fig)  # Clean up
        
        # Display box plot
        st.subheader("Final Fitness Distribution")
        fig = visualizer.plot_fitness_distribution(
            result,
            title=f"Final Fitness Distribution for {result.get('benchmark_name', 'Unknown')}"
        )
        if fig:
            st.pyplot(fig)
            plt.close(fig)  # Clean up
        
    except Exception as e:
        st.error(f"Error displaying comparison details for {selected_file}: {str(e)}")

def create_comparison_table(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create comparison table data from benchmark results.
    
    Args:
        result: Dictionary containing benchmark results
        
    Returns:
        List of dictionaries containing comparison data
    """
    comparison_data = []
    
    if "optimizers" not in result:
        return comparison_data
    
    for opt_name, opt_data in result["optimizers"].items():
        comparison_data.append({
            "Optimizer": opt_name,
            "Best Fitness": opt_data.get("best_fitness", "N/A"),
            "Mean Fitness": opt_data.get("mean_fitness", "N/A"),
            "Std Dev": opt_data.get("std_dev", "N/A"),
            "Mean Runtime": opt_data.get("mean_runtime", "N/A"),
            "Success Rate": opt_data.get("success_rate", "N/A")
        })
    
    return comparison_data 

def find_interactive_reports():
    """Find all interactive HTML reports in the project directory.
    
    Returns:
        List of paths to interactive HTML reports
    """
    import os
    import glob
    from pathlib import Path
    
    # Define directories to search for reports
    search_dirs = [
        "debug_output/reports",
        "output/testing_fixes/moe_validation/reports",
        "tests/debug_output/reports",
        "tests/reports",
        "tests/test_output/reports"
    ]
    
    # Get the project root directory
    project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    
    # Find all interactive reports
    interactive_reports = []
    for search_dir in search_dirs:
        search_path = project_root / search_dir
        if search_path.exists():
            # Look for interactive_report_*.html files
            reports = list(search_path.glob("interactive_report_*.html"))
            # Also check one level deeper
            reports += list(search_path.glob("*/interactive_report_*.html"))
            interactive_reports.extend([str(report) for report in reports])
    
    # Sort by modification time (newest first)
    interactive_reports.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return interactive_reports

def display_interactive_reports(report_paths):
    """Display selected interactive reports for comparison.
    
    Args:
        report_paths: List of paths to interactive HTML reports
    """
    import streamlit as st
    import streamlit.components.v1 as components
    import os
    from pathlib import Path
    import re
    from datetime import datetime
    
    # Create columns for report comparison
    if len(report_paths) > 2:
        st.warning("Comparing more than 2 reports may be difficult to view. Consider selecting fewer reports.")
    
    # Extract metadata from report filenames
    report_metadata = []
    for path in report_paths:
        filename = Path(path).name
        # Extract timestamp from filename (format: interactive_report_YYYY-MM-DD_HH-MM-SS.html)
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                formatted_time = timestamp_str
        else:
            formatted_time = "Unknown time"
        
        report_metadata.append({
            "path": path,
            "filename": filename,
            "timestamp": formatted_time,
            "size": f"{os.path.getsize(path) / 1024:.1f} KB"
        })
    
    # Display metadata table
    st.subheader("Selected Reports")
    metadata_df = pd.DataFrame(report_metadata)
    st.dataframe(metadata_df[["filename", "timestamp", "size"]])
    
    # Create tabs for each report
    report_tabs = st.tabs([f"Report {i+1}: {Path(path).name}" for i, path in enumerate(report_paths)])
    
    # Display each report in its own tab
    for i, (tab, path) in enumerate(zip(report_tabs, report_paths)):
        with tab:
            st.markdown(f"**Viewing: {Path(path).name}**")
            
            # Add button to open in new tab
            file_url = f"file://{path}"
            st.markdown(f"[Open in new tab]({file_url})")
            
            # Display the HTML report
            try:
                with open(path, 'r') as f:
                    html_content = f.read()
                components.html(html_content, height=600, scrolling=True)
            except Exception as e:
                st.error(f"Error loading report: {str(e)}")
    
    # Add side-by-side comparison option for exactly 2 reports
    if len(report_paths) == 2:
        st.subheader("Side-by-Side Comparison")
        st.markdown("This view shows both reports side by side for easier comparison.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{Path(report_paths[0]).name}**")
            try:
                with open(report_paths[0], 'r') as f:
                    html_content = f.read()
                components.html(html_content, height=800, scrolling=True)
            except Exception as e:
                st.error(f"Error loading report: {str(e)}")
        
        with col2:
            st.markdown(f"**{Path(report_paths[1]).name}**")
            try:
                with open(report_paths[1], 'r') as f:
                    html_content = f.read()
                components.html(html_content, height=800, scrolling=True)
            except Exception as e:
                st.error(f"Error loading report: {str(e)}")