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
    
    st.header("Optimizer Comparison")
    
    if not result_files:
        st.warning("No benchmark results found.")
        return
    
    # Filter for comparison results
    comparison_files = [f for f in result_files if is_comparison_result(f, visualizer)]
    
    if not comparison_files:
        st.warning("No comparison results found. Run a comparison benchmark first.")
        return
    
    # Select comparison result to view
    selected_file = st.selectbox(
        "Select a comparison result to view",
        comparison_files
    )
    
    if selected_file:
        display_comparison_details(selected_file, visualizer)

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