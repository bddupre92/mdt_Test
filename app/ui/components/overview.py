"""
Overview Component

This module provides the overview page component for the benchmark dashboard.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def render_overview(results_dir: str, result_files: List[str], visualizer):
    """Render the overview page.
    
    Args:
        results_dir: Directory containing benchmark results
        result_files: List of available result files
        visualizer: Instance of BenchmarkVisualizer
    """
    import streamlit as st  # Import inside function
    
    st.header("Benchmark Results Overview")
    
    if not result_files:
        st.warning("No benchmark results found.")
        return
    
    # Display summary of available benchmarks
    st.subheader("Available Benchmark Results")
    
    # Create summary table
    summary_data = create_summary_table(result_files, visualizer)
    
    # Display summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
        
        # Select a benchmark result to view
        selected_file = st.selectbox(
            "Select a benchmark result to view",
            result_files
        )
        
        if selected_file:
            display_benchmark_details(selected_file, visualizer)
    else:
        st.info("No valid benchmark results found to display.")

def create_summary_table(result_files: List[str], visualizer) -> List[Dict[str, Any]]:
    """Create summary table data from benchmark results.
    
    Args:
        result_files: List of result files
        visualizer: Instance of BenchmarkVisualizer
        
    Returns:
        List of dictionaries containing summary data
    """
    import streamlit as st  # Import inside function
    
    summary_data = []
    
    for filename in result_files:
        try:
            result = visualizer.load_results(filename)
            
            # Extract basic info
            benchmark_name = result.get("benchmark_name", "Unknown")
            timestamp = result.get("timestamp", "Unknown")
            
            # Check if it's a comparison or single optimizer benchmark
            if "optimizers" in result:
                optimizer_count = len(result["optimizers"])
                best_optimizer = "N/A"
                best_fitness = float('inf')
                
                for opt_name, opt_data in result["optimizers"].items():
                    if "best_fitness" in opt_data and opt_data["best_fitness"] < best_fitness:
                        best_fitness = opt_data["best_fitness"]
                        best_optimizer = opt_name
                
                summary_data.append({
                    "Filename": filename,
                    "Benchmark": benchmark_name,
                    "Type": "Comparison",
                    "Optimizers": optimizer_count,
                    "Best Optimizer": best_optimizer,
                    "Best Fitness": best_fitness,
                    "Timestamp": timestamp
                })
            else:
                optimizer_id = result.get("optimizer_id", "Unknown")
                best_fitness = result.get("best_fitness", "N/A")
                
                summary_data.append({
                    "Filename": filename,
                    "Benchmark": benchmark_name,
                    "Type": "Single",
                    "Optimizers": 1,
                    "Best Optimizer": optimizer_id,
                    "Best Fitness": best_fitness,
                    "Timestamp": timestamp
                })
        
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")
    
    return summary_data

def display_benchmark_details(selected_file: str, visualizer):
    """Display detailed information for a selected benchmark.
    
    Args:
        selected_file: Name of the selected result file
        visualizer: Instance of BenchmarkVisualizer
    """
    import streamlit as st  # Import inside function
    
    st.subheader(f"Details for {selected_file}")
    
    try:
        result = visualizer.load_results(selected_file)
        
        # Display basic information
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Benchmark Function:** {result.get('benchmark_name', 'Unknown')}")
            st.write(f"**Dimension:** {result.get('dimension', 'Unknown')}")
            st.write(f"**Bounds:** {result.get('bounds', 'Unknown')}")
        
        with col2:
            st.write(f"**Max Evaluations:** {result.get('max_evaluations', 'Unknown')}")
            st.write(f"**Number of Runs:** {result.get('num_runs', 'Unknown')}")
            st.write(f"**Timestamp:** {result.get('timestamp', 'Unknown')}")
        
        # Display convergence plot
        st.subheader("Convergence Plot")
        fig = visualizer.plot_convergence(
            result,
            title=f"Convergence for {result.get('benchmark_name', 'Unknown')}"
        )
        if fig:
            st.pyplot(fig)
            plt.close(fig)  # Clean up
        
    except Exception as e:
        st.error(f"Error displaying details for {selected_file}: {str(e)}") 