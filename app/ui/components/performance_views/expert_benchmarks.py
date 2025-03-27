"""
Expert Benchmarks View for Performance Analysis

This module provides UI components for analyzing the performance of individual expert models,
enabling detailed comparison and benchmarking of different experts in the MoE system.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import logging

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

logger = logging.getLogger(__name__)

def render_expert_benchmarks(expert_benchmarks: Dict[str, Any]):
    """
    Render the Expert Benchmarks view for detailed analysis of individual expert models.
    
    Args:
        expert_benchmarks: Dictionary containing benchmark metrics for each expert model
    """
    st.header("Expert Model Benchmarks")
    
    st.markdown("""
    This view provides detailed metrics for each individual expert model, allowing 
    benchmarking and comparative analysis of the experts within your MoE system.
    """)
    
    if not expert_benchmarks:
        st.warning("No expert benchmark data available")
        
        st.markdown("""
        ### Getting Started with Expert Benchmarking
        
        To generate expert benchmarks:
        
        1. Use the PerformanceAnalyzer to evaluate each expert individually
        2. The system will generate metrics for each expert model
        3. Results will be saved to the system state
        
        Example code:
        ```python
        from moe_framework.analysis.performance_analyzer import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        expert_metrics = analyzer.benchmark_experts(moe_system, test_data)
        
        # Update the system state
        system_state.performance_metrics["expert_benchmarks"] = expert_metrics
        state_manager.save_state(system_state, "path/to/checkpoint")
        ```
        """)
        return
    
    # Create a DataFrame of expert metrics for comparison
    expert_metrics_df = create_expert_metrics_dataframe(expert_benchmarks)
    
    # Create tabs for different analysis views
    expert_tabs = st.tabs([
        "Performance Comparison", 
        "Detailed Metrics", 
        "Visualizations",
        "Expert Analysis"
    ])
    
    with expert_tabs[0]:
        render_performance_comparison(expert_metrics_df, expert_benchmarks)
    
    with expert_tabs[1]:
        render_detailed_metrics(expert_metrics_df, expert_benchmarks)
    
    with expert_tabs[2]:
        render_expert_visualizations(expert_benchmarks)
    
    with expert_tabs[3]:
        render_expert_analysis(expert_benchmarks)

def create_expert_metrics_dataframe(expert_benchmarks: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a DataFrame of expert metrics for comparison.
    
    Args:
        expert_benchmarks: Dictionary containing benchmark metrics for each expert model
        
    Returns:
        DataFrame with expert metrics
    """
    # Extract common metrics for each expert
    expert_data = {}
    
    for expert_name, metrics in expert_benchmarks.items():
        if isinstance(metrics, dict):
            expert_data[expert_name] = {
                "RMSE": metrics.get("rmse", np.nan),
                "MAE": metrics.get("mae", np.nan),
                "R²": metrics.get("r2", np.nan),
                "Training Time (s)": metrics.get("training_time", np.nan),
                "Inference Time (s)": metrics.get("inference_time", np.nan),
                "Sample Count": metrics.get("sample_count", np.nan)
            }
            
            # Add additional metrics if available
            if "domain_specific_metrics" in metrics and isinstance(metrics["domain_specific_metrics"], dict):
                for metric_name, value in metrics["domain_specific_metrics"].items():
                    expert_data[expert_name][metric_name] = value
    
    # Create DataFrame
    df = pd.DataFrame(expert_data).T
    
    # Calculate ranks
    if not df.empty and len(df) > 1:
        # Lower is better for RMSE, MAE, training time, inference time
        for col in ["RMSE", "MAE", "Training Time (s)", "Inference Time (s)"]:
            if col in df.columns:
                df[f"{col} Rank"] = df[col].rank()
        
        # Higher is better for R²
        if "R²" in df.columns:
            df["R² Rank"] = df["R²"].rank(ascending=False)
    
    return df

def render_performance_comparison(expert_df: pd.DataFrame, expert_benchmarks: Dict[str, Any]):
    """
    Render performance comparison view for expert models.
    
    Args:
        expert_df: DataFrame with expert metrics
        expert_benchmarks: Dictionary containing benchmark metrics for each expert model
    """
    st.subheader("Expert Performance Comparison")
    
    if expert_df.empty:
        st.info("No expert performance data available for comparison")
        return
    
    # Display the dataframe with metrics
    # Format the dataframe for display
    display_columns = [col for col in expert_df.columns if not col.endswith(" Rank")]
    formatted_df = expert_df[display_columns].applymap(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
    )
    
    st.dataframe(formatted_df)
    
    # Visualize key metrics
    st.subheader("Key Metrics Comparison")
    
    # Create metrics comparison charts
    metrics_to_plot = [col for col in ["RMSE", "MAE", "R²"] if col in expert_df.columns]
    
    if metrics_to_plot:
        # Create a figure with subplots based on number of metrics
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 6))
        
        # Handle the case where there's only one metric
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics_to_plot):
            expert_names = expert_df.index.tolist()
            metric_values = expert_df[metric].values
            
            # Create the bar chart
            bars = axes[i].bar(expert_names, metric_values)
            axes[i].set_title(f"{metric} by Expert")
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    axes[i].text(
                        bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0
                    )
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Display performance comparison summary
    st.subheader("Performance Ranking")
    
    rank_columns = [col for col in expert_df.columns if col.endswith(" Rank")]
    if rank_columns:
        # Get the overall ranking
        expert_df["Overall Rank"] = expert_df[rank_columns].mean(axis=1)
        
        # Sort by overall rank
        sorted_df = expert_df.sort_values("Overall Rank")
        
        # Display the rankings
        rank_display = sorted_df[["Overall Rank"] + rank_columns].applymap(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
        )
        
        st.dataframe(rank_display)
        
        # Create a summary
        best_expert = sorted_df.index[0]
        st.markdown(f"**Best Overall Expert**: `{best_expert}`")
        
        # Check for best in specific metrics
        for metric in ["RMSE", "MAE", "R²"]:
            if f"{metric} Rank" in rank_columns:
                best_for_metric = expert_df.sort_values(f"{metric} Rank").index[0]
                if metric == "R²":
                    # For R², higher is better, so we sort in reverse
                    best_for_metric = expert_df.sort_values(f"{metric} Rank").index[0]
                st.markdown(f"**Best for {metric}**: `{best_for_metric}`")

def render_detailed_metrics(expert_df: pd.DataFrame, expert_benchmarks: Dict[str, Any]):
    """
    Render detailed metrics view for expert models.
    
    Args:
        expert_df: DataFrame with expert metrics
        expert_benchmarks: Dictionary containing benchmark metrics for each expert model
    """
    st.subheader("Detailed Expert Metrics")
    
    if not expert_benchmarks:
        st.info("No detailed expert metrics available")
        return
    
    # Create expert selector
    expert_names = list(expert_benchmarks.keys())
    if not expert_names:
        st.info("No experts found in benchmark data")
        return
    
    selected_expert = st.selectbox("Select Expert for Detailed View", expert_names)
    
    if selected_expert not in expert_benchmarks:
        st.error(f"Expert '{selected_expert}' not found in benchmark data")
        return
    
    # Get metrics for selected expert
    expert_metrics = expert_benchmarks[selected_expert]
    
    if not isinstance(expert_metrics, dict):
        st.error(f"Invalid metrics data for expert '{selected_expert}'")
        return
    
    # Display metrics in expandable sections
    with st.expander("Basic Metrics", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rmse = expert_metrics.get("rmse", "N/A")
            st.metric("RMSE", f"{rmse:.4f}" if isinstance(rmse, (int, float)) else rmse)
        
        with col2:
            mae = expert_metrics.get("mae", "N/A")
            st.metric("MAE", f"{mae:.4f}" if isinstance(mae, (int, float)) else mae)
        
        with col3:
            r2 = expert_metrics.get("r2", "N/A")
            st.metric("R²", f"{r2:.4f}" if isinstance(r2, (int, float)) else r2)
    
    with st.expander("Performance Metrics", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_time = expert_metrics.get("training_time", "N/A")
            st.metric(
                "Training Time", 
                f"{train_time:.2f} s" if isinstance(train_time, (int, float)) else train_time
            )
        
        with col2:
            inference_time = expert_metrics.get("inference_time", "N/A")
            st.metric(
                "Inference Time", 
                f"{inference_time:.4f} s" if isinstance(inference_time, (int, float)) else inference_time
            )
        
        with col3:
            memory = expert_metrics.get("memory_usage", "N/A")
            st.metric(
                "Memory Usage", 
                f"{memory:.2f} MB" if isinstance(memory, (int, float)) else memory
            )
    
    # Display domain-specific metrics if available
    if "domain_specific_metrics" in expert_metrics and isinstance(expert_metrics["domain_specific_metrics"], dict):
        with st.expander("Domain-Specific Metrics", expanded=True):
            domain_metrics = expert_metrics["domain_specific_metrics"]
            
            # Create columns based on number of metrics
            num_metrics = len(domain_metrics)
            cols_per_row = 3
            num_rows = (num_metrics + cols_per_row - 1) // cols_per_row
            
            for row in range(num_rows):
                # Create columns for this row
                cols = st.columns(cols_per_row)
                
                # Fill the columns with metrics
                for col_idx in range(cols_per_row):
                    metric_idx = row * cols_per_row + col_idx
                    
                    if metric_idx < num_metrics:
                        metric_name = list(domain_metrics.keys())[metric_idx]
                        metric_value = domain_metrics[metric_name]
                        
                        with cols[col_idx]:
                            # Format the value based on type
                            if isinstance(metric_value, (int, float)):
                                formatted_value = f"{metric_value:.4f}"
                            else:
                                formatted_value = str(metric_value)
                            
                            st.metric(metric_name, formatted_value)
    
    # Display error distribution if available
    if "error_distribution" in expert_metrics and isinstance(expert_metrics["error_distribution"], dict):
        with st.expander("Error Distribution", expanded=True):
            error_dist = expert_metrics["error_distribution"]
            
            # Extract distribution data
            if "bins" in error_dist and "counts" in error_dist:
                bins = error_dist["bins"]
                counts = error_dist["counts"]
                
                # Create histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(bins[:-1], counts, width=bins[1]-bins[0], align="edge")
                ax.set_title(f"Error Distribution for {selected_expert}")
                ax.set_xlabel("Error")
                ax.set_ylabel("Count")
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Error distribution data is not in the expected format")
    
    # Display feature importance if available
    if "feature_importance" in expert_metrics and isinstance(expert_metrics["feature_importance"], dict):
        with st.expander("Feature Importance", expanded=True):
            feature_imp = expert_metrics["feature_importance"]
            
            # Convert to DataFrame for display
            imp_df = pd.DataFrame({
                "Feature": list(feature_imp.keys()),
                "Importance": list(feature_imp.values())
            })
            
            # Sort by importance
            imp_df = imp_df.sort_values("Importance", ascending=False)
            
            # Display as table
            st.dataframe(imp_df)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, max(6, len(feature_imp) * 0.3)))
            bars = ax.barh(imp_df["Feature"], imp_df["Importance"])
            ax.set_title(f"Feature Importance for {selected_expert}")
            ax.set_xlabel("Importance")
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.4f}',
                    ha='left', va='center'
                )
            
            plt.tight_layout()
            st.pyplot(fig)

def render_expert_visualizations(expert_benchmarks: Dict[str, Any]):
    """
    Render expert visualization view.
    
    Args:
        expert_benchmarks: Dictionary containing benchmark metrics for each expert model
    """
    st.subheader("Expert Visualizations")
    
    if not expert_benchmarks:
        st.info("No expert benchmark data available for visualization")
        return
    
    # Create expert selector
    expert_names = list(expert_benchmarks.keys())
    if not expert_names:
        st.info("No experts found in benchmark data")
        return
    
    # Option to select multiple experts for comparison
    selected_experts = st.multiselect(
        "Select Experts for Visualization", 
        expert_names,
        default=expert_names[:min(3, len(expert_names))]
    )
    
    if not selected_experts:
        st.info("Please select at least one expert to visualize")
        return
    
    # Check if prediction vs actual data is available
    has_pred_actual = any(
        "predictions" in expert_benchmarks[expert] and "actual" in expert_benchmarks[expert]
        for expert in selected_experts
    )
    
    # Create visualization selector
    viz_options = [
        "Performance Metrics",
        "Training vs Inference Time"
    ]
    
    if has_pred_actual:
        viz_options.append("Predictions vs Actual")
    
    selected_viz = st.selectbox("Select Visualization", viz_options)
    
    # Render the selected visualization
    if selected_viz == "Performance Metrics":
        render_performance_metrics_viz(selected_experts, expert_benchmarks)
    elif selected_viz == "Training vs Inference Time":
        render_time_metrics_viz(selected_experts, expert_benchmarks)
    elif selected_viz == "Predictions vs Actual":
        render_pred_actual_viz(selected_experts, expert_benchmarks)

def render_performance_metrics_viz(selected_experts: List[str], expert_benchmarks: Dict[str, Any]):
    """
    Render performance metrics visualization for selected experts.
    
    Args:
        selected_experts: List of expert names to include in visualization
        expert_benchmarks: Dictionary containing benchmark metrics for each expert model
    """
    # Create metrics data for selected experts
    metrics_data = {
        "Expert": [],
        "Metric": [],
        "Value": []
    }
    
    for expert in selected_experts:
        if expert in expert_benchmarks and isinstance(expert_benchmarks[expert], dict):
            metrics = expert_benchmarks[expert]
            
            for metric_name in ["rmse", "mae", "r2"]:
                if metric_name in metrics:
                    metrics_data["Expert"].append(expert)
                    metrics_data["Metric"].append(metric_name.upper())
                    metrics_data["Value"].append(metrics[metric_name])
    
    if not metrics_data["Expert"]:
        st.info("No metrics data available for selected experts")
        return
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot grouped bars with seaborn
    sns.barplot(x="Expert", y="Value", hue="Metric", data=metrics_df, ax=ax)
    
    ax.set_title("Performance Metrics Comparison")
    ax.set_ylabel("Value")
    ax.legend(title="Metric")
    
    plt.tight_layout()
    st.pyplot(fig)

def render_time_metrics_viz(selected_experts: List[str], expert_benchmarks: Dict[str, Any]):
    """
    Render time metrics visualization for selected experts.
    
    Args:
        selected_experts: List of expert names to include in visualization
        expert_benchmarks: Dictionary containing benchmark metrics for each expert model
    """
    # Create time metrics data for selected experts
    time_data = {
        "Expert": [],
        "Training Time (s)": [],
        "Inference Time (s)": []
    }
    
    for expert in selected_experts:
        if expert in expert_benchmarks and isinstance(expert_benchmarks[expert], dict):
            metrics = expert_benchmarks[expert]
            
            if "training_time" in metrics or "inference_time" in metrics:
                time_data["Expert"].append(expert)
                time_data["Training Time (s)"].append(metrics.get("training_time", np.nan))
                time_data["Inference Time (s)"].append(metrics.get("inference_time", np.nan))
    
    if not time_data["Expert"]:
        st.info("No time metrics available for selected experts")
        return
    
    # Convert to DataFrame
    time_df = pd.DataFrame(time_data)
    
    # Melt the DataFrame for seaborn
    melted_df = pd.melt(
        time_df, 
        id_vars=["Expert"],
        value_vars=["Training Time (s)", "Inference Time (s)"],
        var_name="Metric",
        value_name="Time (s)"
    )
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot grouped bars with seaborn
    sns.barplot(x="Expert", y="Time (s)", hue="Metric", data=melted_df, ax=ax)
    
    ax.set_title("Training vs Inference Time")
    ax.set_ylabel("Time (seconds)")
    ax.legend(title="Time Metric")
    
    # Use logarithmic scale if values differ by orders of magnitude
    if time_df["Training Time (s)"].max() / time_df["Inference Time (s)"].min() > 100:
        ax.set_yscale("log")
        ax.set_ylabel("Time (seconds, log scale)")
    
    plt.tight_layout()
    st.pyplot(fig)

def render_pred_actual_viz(selected_experts: List[str], expert_benchmarks: Dict[str, Any]):
    """
    Render predictions vs actual visualization for selected experts.
    
    Args:
        selected_experts: List of expert names to include in visualization
        expert_benchmarks: Dictionary containing benchmark metrics for each expert model
    """
    # Check which experts have prediction vs actual data
    valid_experts = []
    for expert in selected_experts:
        if expert in expert_benchmarks and isinstance(expert_benchmarks[expert], dict):
            metrics = expert_benchmarks[expert]
            if "predictions" in metrics and "actual" in metrics:
                valid_experts.append(expert)
    
    if not valid_experts:
        st.info("No prediction vs actual data available for selected experts")
        return
    
    # Limit to max 4 experts for visualization clarity
    if len(valid_experts) > 4:
        st.warning(f"Too many experts selected for clear visualization. Showing first 4 of {len(valid_experts)} selected.")
        valid_experts = valid_experts[:4]
    
    # Create a grid of scatter plots
    n_experts = len(valid_experts)
    n_cols = min(2, n_experts)
    n_rows = (n_experts + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    
    # Handle the case where axes is a single Axes object or 1D array
    if n_experts == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Create scatter plots
    for i, expert in enumerate(valid_experts):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        metrics = expert_benchmarks[expert]
        predictions = metrics["predictions"]
        actual = metrics["actual"]
        
        # Create scatter plot
        ax.scatter(actual, predictions, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(min(predictions), min(actual))
        max_val = max(max(predictions), max(actual))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title(f"{expert}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        
        # Add RMSE to plot
        rmse = metrics.get("rmse", np.nan)
        if not np.isnan(rmse):
            ax.text(
                0.05, 0.95, f"RMSE: {rmse:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", alpha=0.1)
            )
    
    # Hide unused subplots
    for i in range(n_experts, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    st.pyplot(fig)

def render_expert_analysis(expert_benchmarks: Dict[str, Any]):
    """
    Render expert analysis view with recommendations and insights.
    
    Args:
        expert_benchmarks: Dictionary containing benchmark metrics for each expert model
    """
    st.subheader("Expert Analysis & Recommendations")
    
    if not expert_benchmarks:
        st.info("No expert data available for analysis")
        return
    
    # Create DataFrame for analysis
    expert_data = {}
    
    for expert_name, metrics in expert_benchmarks.items():
        if isinstance(metrics, dict):
            expert_data[expert_name] = {
                "RMSE": metrics.get("rmse", np.nan),
                "MAE": metrics.get("mae", np.nan),
                "R²": metrics.get("r2", np.nan),
                "Training Time (s)": metrics.get("training_time", np.nan),
                "Inference Time (s)": metrics.get("inference_time", np.nan)
            }
    
    # Convert to DataFrame
    df = pd.DataFrame(expert_data).T
    
    # Generate insights
    insights = []
    
    # Find best performing expert
    if "RMSE" in df.columns and not df["RMSE"].isnull().all():
        best_rmse_expert = df["RMSE"].idxmin()
        insights.append(f"✅ `{best_rmse_expert}` has the lowest RMSE ({df.loc[best_rmse_expert, 'RMSE']:.4f}), making it the most accurate expert for prediction.")
    
    # Find fastest expert for inference
    if "Inference Time (s)" in df.columns and not df["Inference Time (s)"].isnull().all():
        fastest_expert = df["Inference Time (s)"].idxmin()
        insights.append(f"⚡ `{fastest_expert}` has the fastest inference time ({df.loc[fastest_expert, 'Inference Time (s)']:.4f}s), making it optimal for real-time applications.")
    
    # Check for experts with good accuracy-speed tradeoff
    if "RMSE" in df.columns and "Inference Time (s)" in df.columns:
        valid_experts = df.dropna(subset=["RMSE", "Inference Time (s)"]).index.tolist()
        
        if valid_experts:
            # Calculate normalized scores (lower is better for both)
            valid_df = df.loc[valid_experts, ["RMSE", "Inference Time (s)"]]
            normalized_df = valid_df / valid_df.max()
            
            # Calculate balanced score (equal weight to accuracy and speed)
            balanced_scores = normalized_df.mean(axis=1)
            best_balanced = balanced_scores.idxmin()
            
            insights.append(f"🔄 `{best_balanced}` offers the best balance between accuracy and speed.")
    
    # Check for potential issues
    for expert, row in df.iterrows():
        # Check for significantly worse performance
        if "RMSE" in df.columns and not np.isnan(row["RMSE"]):
            rmse_threshold = df["RMSE"].median() * 1.5
            if row["RMSE"] > rmse_threshold:
                insights.append(f"⚠️ `{expert}` has significantly higher RMSE ({row['RMSE']:.4f}) than the median, suggesting potential issues.")
        
        # Check for very slow training time
        if "Training Time (s)" in df.columns and not np.isnan(row["Training Time (s)"]):
            time_threshold = df["Training Time (s)"].median() * 2
            if row["Training Time (s)"] > time_threshold:
                insights.append(f"⚠️ `{expert}` has unusually long training time ({row['Training Time (s)']:.2f}s), which may indicate optimization opportunities.")
    
    # Display insights
    if insights:
        st.markdown("### Key Insights")
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("No significant insights available from the current benchmark data")
    
    # Generate recommendations
    st.markdown("### Recommendations")
    
    recommendations = []
    
    # Recommend for gating network
    if "RMSE" in df.columns and not df["RMSE"].isnull().all():
        domain_experts = []
        for expert in df.index:
            if "_" in expert:
                domain = expert.split("_")[0]
                domain_experts.append((domain, expert))
        
        if domain_experts:
            recommendations.append("Consider configuring the gating network to favor domain-specific experts for their respective data types:")
            
            for domain, expert in domain_experts:
                recommendations.append(f"- For {domain} data, bias weights toward `{expert}`")
    
    # Recommend for resource constraints
    if "Inference Time (s)" in df.columns and "RMSE" in df.columns:
        valid_df = df.dropna(subset=["RMSE", "Inference Time (s)"])
        
        if not valid_df.empty:
            fast_expert = valid_df["Inference Time (s)"].idxmin()
            recommendations.append(f"For resource-constrained environments (e.g., mobile devices), consider using `{fast_expert}` as the primary expert.")
    
    # Recommend for accuracy-critical applications
    if "RMSE" in df.columns:
        valid_df = df.dropna(subset=["RMSE"])
        
        if not valid_df.empty:
            accurate_expert = valid_df["RMSE"].idxmin()
            recommendations.append(f"For accuracy-critical applications, prioritize `{accurate_expert}` in the expert ensemble.")
    
    # Display recommendations
    if recommendations:
        for recommendation in recommendations:
            st.markdown(recommendation)
    else:
        st.info("No specific recommendations available based on the current benchmark data")
