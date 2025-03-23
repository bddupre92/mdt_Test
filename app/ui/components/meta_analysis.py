"""
Meta Analysis Component

This module provides the meta-optimizer analysis page component for the benchmark dashboard.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def render_meta_analysis(results_dir: str, result_files: List[str], visualizer):
    """Render the meta-optimizer analysis page.
    
    Args:
        results_dir: Directory containing benchmark results
        result_files: List of available result files
        visualizer: Instance of BenchmarkVisualizer
    """
    import streamlit as st  # Import inside function
    
    st.header("Meta-Optimizer Analysis")
    
    if not result_files:
        st.warning("No benchmark results found.")
        return
    
    # Filter for meta-optimizer results
    meta_files = [f for f in result_files if is_meta_result(f, visualizer)]
    
    if not meta_files:
        st.warning("No meta-optimizer results found. Run a meta-optimizer benchmark first.")
        return
    
    # Select meta-optimizer result to view
    selected_file = st.selectbox(
        "Select a meta-optimizer result to view",
        meta_files
    )
    
    if selected_file:
        display_meta_analysis(selected_file, visualizer)

def is_meta_result(filename: str, visualizer) -> bool:
    """Check if a result file contains meta-optimizer data.
    
    Args:
        filename: Name of the result file
        visualizer: Instance of BenchmarkVisualizer
        
    Returns:
        True if file contains meta-optimizer data, False otherwise
    """
    try:
        result = visualizer.load_results(filename)
        return "meta_optimizer" in result
    except:
        return False

def display_meta_analysis(selected_file: str, visualizer):
    """Display detailed meta-optimizer analysis.
    
    Args:
        selected_file: Name of the selected result file
        visualizer: Instance of BenchmarkVisualizer
    """
    import streamlit as st  # Import inside function
    
    try:
        result = visualizer.load_results(selected_file)
        
        # Display basic information
        st.subheader("Meta-Optimizer Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Meta-Optimizer:** {result.get('meta_optimizer', {}).get('name', 'Unknown')}")
            st.write(f"**Base Optimizer:** {result.get('base_optimizer', 'Unknown')}")
            st.write(f"**Benchmark Function:** {result.get('benchmark_name', 'Unknown')}")
        
        with col2:
            st.write(f"**Dimension:** {result.get('dimension', 'Unknown')}")
            st.write(f"**Max Evaluations:** {result.get('max_evaluations', 'Unknown')}")
            st.write(f"**Number of Runs:** {result.get('num_runs', 'Unknown')}")
        
        # Display meta-optimizer performance
        st.subheader("Meta-Optimizer Performance")
        meta_data = create_meta_table(result)
        if meta_data:
            meta_df = pd.DataFrame(meta_data)
            st.dataframe(meta_df)
        
        # Display convergence plot
        st.subheader("Meta-Optimizer Convergence")
        fig = visualizer.plot_meta_convergence(
            result,
            title=f"Meta-Optimizer Convergence for {result.get('benchmark_name', 'Unknown')}"
        )
        if fig:
            st.pyplot(fig)
            plt.close(fig)  # Clean up
        
        # Display parameter evolution
        st.subheader("Parameter Evolution")
        fig = visualizer.plot_parameter_evolution(
            result,
            title=f"Parameter Evolution for {result.get('meta_optimizer', {}).get('name', 'Unknown')}"
        )
        if fig:
            st.pyplot(fig)
            plt.close(fig)  # Clean up
        
    except Exception as e:
        st.error(f"Error displaying meta-optimizer analysis for {selected_file}: {str(e)}")

def create_meta_table(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create meta-optimizer performance table data.
    
    Args:
        result: Dictionary containing benchmark results
        
    Returns:
        List of dictionaries containing meta-optimizer performance data
    """
    meta_data = []
    
    if "meta_optimizer" not in result:
        return meta_data
    
    meta_result = result["meta_optimizer"]
    meta_data.append({
        "Best Fitness": meta_result.get("best_fitness", "N/A"),
        "Mean Fitness": meta_result.get("mean_fitness", "N/A"),
        "Std Dev": meta_result.get("std_dev", "N/A"),
        "Mean Runtime": meta_result.get("mean_runtime", "N/A"),
        "Success Rate": meta_result.get("success_rate", "N/A"),
        "Best Parameters": str(meta_result.get("best_parameters", "N/A"))
    })
    
    return meta_data 