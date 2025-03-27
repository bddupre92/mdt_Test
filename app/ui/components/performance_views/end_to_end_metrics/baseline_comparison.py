"""
Baseline Comparison Component for End-to-End Performance Metrics

This module provides visualization and analysis of MoE performance compared to baseline models.
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

from app.ui.components.performance_views.helpers import safe_get_metric

def render_baseline_comparison(end_to_end_metrics: Dict[str, Any]):
    """
    Render the baseline comparison component for end-to-end performance metrics.
    
    Args:
        end_to_end_metrics: Dictionary containing end-to-end performance metrics
    """
    st.subheader("Baseline Comparison Analysis")
    
    # Extract baseline comparison data
    baseline_comparisons = safe_get_metric(end_to_end_metrics, "baseline_comparisons", {})
    
    if not baseline_comparisons or not isinstance(baseline_comparisons, dict):
        st.info("No baseline comparison data available")
        
        st.markdown("""
        ### Getting Started with Baseline Comparisons
        
        To generate baseline comparison data:
        
        1. Run your MoE model along with baseline models on the same test dataset
        2. Calculate metrics for each model using the same evaluation criteria
        3. Use the comparison functions in the MoEMetricsCalculator
        
        Example code:
        ```python
        from baseline_comparison.moe_metrics import MoEMetricsCalculator
        
        calculator = MoEMetricsCalculator()
        comparisons = calculator.compare_with_baselines(
            moe_predictions=moe_preds,
            baseline_predictions={
                "linear_regression": lr_preds,
                "random_forest": rf_preds
            },
            y_true=y_test
        )
        
        # Update the system state
        system_state.performance_metrics["end_to_end_metrics"]["baseline_comparisons"] = comparisons
        state_manager.save_state(system_state, "path/to/checkpoint")
        ```
        """)
        return
    
    # Get available baseline models
    baseline_models = list(baseline_comparisons.keys())
    
    if not baseline_models:
        st.info("No baseline models found in comparison data")
        return
    
    # Create tabs for different comparison views
    comparison_tabs = st.tabs([
        "Overall Comparison",
        "Metric Comparison",
        "Improvement Analysis",
        "Error Analysis"
    ])
    
    with comparison_tabs[0]:
        render_overall_comparison(baseline_comparisons)
    
    with comparison_tabs[1]:
        render_metric_comparison(baseline_comparisons)
    
    with comparison_tabs[2]:
        render_improvement_analysis(baseline_comparisons)
    
    with comparison_tabs[3]:
        render_error_analysis(baseline_comparisons)

def render_overall_comparison(baseline_comparisons: Dict[str, Any]):
    """
    Render overall comparison between MoE and baseline models.
    
    Args:
        baseline_comparisons: Dictionary containing baseline comparison data
    """
    st.subheader("Overall Model Comparison")
    
    # Create a DataFrame for comparison
    comparison_data = []
    
    for model_name, metrics in baseline_comparisons.items():
        if isinstance(metrics, dict):
            model_data = {
                "Model": model_name,
                "RMSE": metrics.get("rmse", np.nan),
                "MAE": metrics.get("mae", np.nan),
                "R²": metrics.get("r2", np.nan),
                "Training Time (s)": metrics.get("training_time", np.nan),
                "Inference Time (s)": metrics.get("inference_time", np.nan)
            }
            
            # Add additional metrics if available
            for key in metrics:
                if key not in model_data and not key.startswith("_"):
                    model_data[key] = metrics[key]
            
            comparison_data.append(model_data)
    
    if not comparison_data:
        st.info("No valid comparison data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Format the table
    formatted_df = df.copy()
    for col in df.columns:
        if col != "Model":
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
            )
    
    # Display the table
    st.dataframe(formatted_df, use_container_width=True)
    
    # Display performance improvements
    if "MoE" in df["Model"].values:
        st.subheader("MoE Performance Improvements")
        
        moe_row = df[df["Model"] == "MoE"].iloc[0]
        other_models = df[df["Model"] != "MoE"]
        
        if not other_models.empty:
            improvements = []
            
            for _, baseline in other_models.iterrows():
                model_name = baseline["Model"]
                
                improvement_data = {"Baseline": model_name}
                
                # Calculate improvements for error metrics (lower is better)
                for metric in ["RMSE", "MAE"]:
                    if metric in df.columns and not pd.isna(moe_row[metric]) and not pd.isna(baseline[metric]):
                        moe_value = moe_row[metric]
                        baseline_value = baseline[metric]
                        
                        if baseline_value != 0:
                            improvement = (baseline_value - moe_value) / baseline_value * 100
                            improvement_data[f"{metric} Improvement"] = improvement
                
                # Calculate improvements for R² (higher is better)
                if "R²" in df.columns and not pd.isna(moe_row["R²"]) and not pd.isna(baseline["R²"]):
                    moe_r2 = moe_row["R²"]
                    baseline_r2 = baseline["R²"]
                    
                    if baseline_r2 != 0:
                        improvement = (moe_r2 - baseline_r2) / abs(baseline_r2) * 100
                        improvement_data["R² Improvement"] = improvement
                
                improvements.append(improvement_data)
            
            # Create improvement DataFrame
            imp_df = pd.DataFrame(improvements)
            
            # Format for display
            formatted_imp_df = imp_df.copy()
            for col in imp_df.columns:
                if col != "Baseline":
                    formatted_imp_df[col] = formatted_imp_df[col].apply(
                        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
                    )
            
            st.dataframe(formatted_imp_df, use_container_width=True)
            
            # Create a bar chart of improvements
            metrics_to_plot = [col for col in imp_df.columns if col.endswith("Improvement")]
            
            if metrics_to_plot:
                # Melt the DataFrame for easier plotting
                plot_df = pd.melt(
                    imp_df,
                    id_vars=["Baseline"],
                    value_vars=metrics_to_plot,
                    var_name="Metric",
                    value_name="Improvement (%)"
                )
                
                # Clean up metric names for display
                plot_df["Metric"] = plot_df["Metric"].str.replace(" Improvement", "")
                
                # Create the chart
                fig, ax = plt.subplots(figsize=(12, 6))
                
                sns.barplot(x="Baseline", y="Improvement (%)", hue="Metric", data=plot_df, ax=ax)
                
                ax.set_title("MoE Improvement Over Baselines (%)")
                ax.set_xlabel("Baseline Model")
                ax.set_ylabel("Improvement (%)")
                
                # Add a horizontal line at 0%
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)

def render_metric_comparison(baseline_comparisons: Dict[str, Any]):
    """
    Render detailed metric comparison between MoE and baseline models.
    
    Args:
        baseline_comparisons: Dictionary containing baseline comparison data
    """
    st.subheader("Detailed Metric Comparison")
    
    # Extract common metrics for comparison
    common_metrics = set()
    for model_name, metrics in baseline_comparisons.items():
        if isinstance(metrics, dict):
            common_metrics.update(metrics.keys())
    
    # Remove private or metadata fields
    common_metrics = [m for m in common_metrics if not m.startswith("_") and m not in 
                     ["predictions", "actual", "error_distribution", "feature_importance"]]
    
    if not common_metrics:
        st.info("No common metrics found for comparison")
        return
    
    # Allow user to select metrics for comparison
    selected_metrics = st.multiselect(
        "Select Metrics for Comparison",
        options=sorted(common_metrics),
        default=["rmse", "mae", "r2"] if all(m in common_metrics for m in ["rmse", "mae", "r2"]) else None
    )
    
    if not selected_metrics:
        st.info("Please select at least one metric for comparison")
        return
    
    # Get models for comparison
    models = list(baseline_comparisons.keys())
    
    if not models:
        st.info("No models found for comparison")
        return
    
    # Create DataFrame for selected metrics
    data = []
    
    for metric in selected_metrics:
        metric_row = {"Metric": metric}
        
        for model in models:
            if model in baseline_comparisons and isinstance(baseline_comparisons[model], dict):
                if metric in baseline_comparisons[model]:
                    metric_row[model] = baseline_comparisons[model][metric]
        
        data.append(metric_row)
    
    if not data:
        st.info("No data available for selected metrics")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Format for display
    formatted_df = df.copy()
    for col in formatted_df.columns:
        if col != "Metric":
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
            )
    
    st.dataframe(formatted_df, use_container_width=True)
    
    # Create visualization
    if len(models) > 0 and len(selected_metrics) > 0:
        # Create a melted DataFrame for visualization
        model_metric_data = []
        
        for model in models:
            if model in baseline_comparisons and isinstance(baseline_comparisons[model], dict):
                for metric in selected_metrics:
                    if metric in baseline_comparisons[model]:
                        model_metric_data.append({
                            "Model": model,
                            "Metric": metric,
                            "Value": baseline_comparisons[model][metric]
                        })
        
        if model_metric_data:
            plot_df = pd.DataFrame(model_metric_data)
            
            # Create a grouped bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            sns.barplot(x="Metric", y="Value", hue="Model", data=plot_df, ax=ax)
            
            ax.set_title("Model Performance by Metric")
            ax.set_xlabel("Metric")
            ax.set_ylabel("Value")
            
            plt.tight_layout()
            st.pyplot(fig)

def render_improvement_analysis(baseline_comparisons: Dict[str, Any]):
    """
    Render analysis of where MoE shows the most improvement over baselines.
    
    Args:
        baseline_comparisons: Dictionary containing baseline comparison data
    """
    st.subheader("Improvement Analysis")
    
    # Check if MoE is in the models
    if "MoE" not in baseline_comparisons:
        st.info("MoE model not found in comparison data")
        return
    
    # Check if detailed comparison data is available
    detailed_comparison = {}
    
    for model_name, metrics in baseline_comparisons.items():
        if model_name != "MoE" and isinstance(metrics, dict):
            if "detailed_comparison" in metrics and isinstance(metrics["detailed_comparison"], dict):
                detailed_comparison[model_name] = metrics["detailed_comparison"]
    
    if not detailed_comparison:
        st.info("No detailed comparison data available for improvement analysis")
        return
    
    # Create improvement overview
    st.markdown("### MoE Improvement Areas")
    
    for baseline, comparison in detailed_comparison.items():
        with st.expander(f"{baseline} vs. MoE", expanded=True):
            # Get improvement by segment if available
            segment_improvement = comparison.get("segment_improvement", {})
            
            if segment_improvement and isinstance(segment_improvement, dict):
                # Allow user to select segment type
                segment_types = list(segment_improvement.keys())
                
                if segment_types:
                    selected_segment = st.selectbox(
                        f"Select Segment Type for {baseline}",
                        options=segment_types
                    )
                    
                    if selected_segment in segment_improvement:
                        segment_data = segment_improvement[selected_segment]
                        
                        if isinstance(segment_data, dict) and segment_data:
                            # Create DataFrame for visualization
                            segments = []
                            improvements = []
                            
                            for segment, improvement in segment_data.items():
                                if isinstance(improvement, (int, float)) and not np.isnan(improvement):
                                    segments.append(segment)
                                    improvements.append(improvement)
                            
                            if segments and improvements:
                                df = pd.DataFrame({
                                    "Segment": segments,
                                    "Improvement (%)": improvements
                                })
                                
                                # Sort by improvement
                                df = df.sort_values("Improvement (%)", ascending=False)
                                
                                # Create bar chart
                                fig, ax = plt.subplots(figsize=(12, min(8, max(4, len(df) * 0.4))))
                                
                                # Use green for positive improvements, red for negative
                                colors = ['green' if x >= 0 else 'red' for x in df["Improvement (%)"]]
                                
                                bars = ax.barh(df["Segment"], df["Improvement (%)"], color=colors)
                                
                                # Add data labels
                                for bar in bars:
                                    width = bar.get_width()
                                    ax.text(
                                        width + (1 if width >= 0 else -1),
                                        bar.get_y() + bar.get_height()/2,
                                        f"{width:.2f}%",
                                        ha='left' if width >= 0 else 'right',
                                        va='center'
                                    )
                                
                                ax.set_title(f"MoE Improvement over {baseline} by {selected_segment}")
                                ax.set_xlabel("Improvement (%)")
                                ax.set_ylabel(selected_segment)
                                
                                # Add vertical line at 0%
                                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Display insights
                                best_segment = df.iloc[0]["Segment"]
                                worst_segment = df.iloc[-1]["Segment"]
                                
                                st.markdown(f"""
                                **Insights:**
                                - MoE shows the greatest improvement over {baseline} in the **{best_segment}** segment ({df.iloc[0]["Improvement (%)"]:.2f}%)
                                - MoE shows the least improvement (or decline) over {baseline} in the **{worst_segment}** segment ({df.iloc[-1]["Improvement (%)"]:.2f}%)
                                """)
                                
                                # Add improvement categories
                                improved = df[df["Improvement (%)"] > 0]
                                declined = df[df["Improvement (%)"] < 0]
                                
                                st.markdown(f"""
                                - MoE outperforms {baseline} in **{len(improved)}** out of **{len(df)}** segments
                                - MoE underperforms {baseline} in **{len(declined)}** out of **{len(df)}** segments
                                """)
            
            # Get feature-based improvement if available
            feature_improvement = comparison.get("feature_improvement", {})
            
            if feature_improvement and isinstance(feature_improvement, dict):
                st.markdown("#### Feature-Based Improvement")
                
                # Create DataFrame for visualization
                features = []
                improvements = []
                
                for feature, improvement in feature_improvement.items():
                    if isinstance(improvement, (int, float)) and not np.isnan(improvement):
                        features.append(feature)
                        improvements.append(improvement)
                
                if features and improvements:
                    df = pd.DataFrame({
                        "Feature": features,
                        "Improvement (%)": improvements
                    })
                    
                    # Sort by improvement
                    df = df.sort_values("Improvement (%)", ascending=False)
                    
                    # Display top and bottom 5 features
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Top 5 Features with Greatest Improvement**")
                        top_df = df.head(5)
                        formatted_top = top_df.copy()
                        formatted_top["Improvement (%)"] = formatted_top["Improvement (%)"].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(formatted_top)
                    
                    with col2:
                        st.markdown("**Bottom 5 Features with Least Improvement**")
                        bottom_df = df.tail(5).sort_values("Improvement (%)")
                        formatted_bottom = bottom_df.copy()
                        formatted_bottom["Improvement (%)"] = formatted_bottom["Improvement (%)"].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(formatted_bottom)

def render_error_analysis(baseline_comparisons: Dict[str, Any]):
    """
    Render error analysis comparing MoE and baseline models.
    
    Args:
        baseline_comparisons: Dictionary containing baseline comparison data
    """
    st.subheader("Error Analysis Comparison")
    
    # Check if error distribution data is available
    has_error_data = False
    models_with_errors = []
    
    for model_name, metrics in baseline_comparisons.items():
        if isinstance(metrics, dict) and "error_distribution" in metrics:
            has_error_data = True
            models_with_errors.append(model_name)
    
    if not has_error_data:
        st.info("No error distribution data available for comparison")
        return
    
    # Allow user to select models for comparison
    selected_models = st.multiselect(
        "Select Models for Error Comparison",
        options=models_with_errors,
        default=["MoE"] if "MoE" in models_with_errors else None
    )
    
    if not selected_models:
        st.info("Please select at least one model for error analysis")
        return
    
    # Create error distribution comparison
    error_data = {}
    
    for model in selected_models:
        if model in baseline_comparisons and isinstance(baseline_comparisons[model], dict):
            error_dist = baseline_comparisons[model].get("error_distribution", {})
            
            if isinstance(error_dist, dict):
                if "errors" in error_dist:
                    error_data[model] = error_dist["errors"]
                elif "bins" in error_dist and "counts" in error_dist:
                    # Approximate errors from bins and counts
                    bins = error_dist["bins"]
                    counts = error_dist["counts"]
                    
                    if len(bins) > 1 and len(counts) > 0:
                        # Create representative errors (bin centers weighted by counts)
                        approx_errors = []
                        
                        for i in range(len(counts)):
                            bin_center = (bins[i] + bins[i+1]) / 2 if i < len(bins) - 1 else bins[i]
                            approx_errors.extend([bin_center] * counts[i])
                        
                        error_data[model] = approx_errors
    
    if not error_data:
        st.info("No valid error data found for selected models")
        return
    
    # Create error distribution visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model, errors in error_data.items():
        if errors:
            sns.kdeplot(errors, label=model, ax=ax)
    
    ax.set_title("Error Distribution Comparison")
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display error statistics comparison
    st.subheader("Error Statistics Comparison")
    
    stats_data = []
    
    for model, errors in error_data.items():
        if errors:
            stats_data.append({
                "Model": model,
                "Mean Error": np.mean(errors),
                "Median Error": np.median(errors),
                "Error Std Dev": np.std(errors),
                "Mean Abs Error": np.mean(np.abs(errors)),
                "Max Error": np.max(np.abs(errors))
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        # Format for display
        formatted_stats = stats_df.copy()
        for col in stats_df.columns:
            if col != "Model":
                formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(formatted_stats, use_container_width=True)
        
        # Create a radar chart for error metrics comparison
        if len(stats_df) > 1:
            # Normalize values for radar chart
            radar_metrics = ["Mean Abs Error", "Error Std Dev", "Max Error"]
            radar_df = stats_df[["Model"] + radar_metrics].copy()
            
            # Normalize each metric (lower is better, so smaller normalized value is better)
            for metric in radar_metrics:
                max_val = radar_df[metric].max()
                if max_val > 0:
                    radar_df[metric] = radar_df[metric] / max_val
            
            # Create radar chart
            st.markdown("### Error Metrics Comparison (Normalized)")
            
            # Number of variables
            categories = radar_metrics
            N = len(categories)
            
            # Create angles for radar chart
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create subplot with polar projection
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            # Draw one axis per variable and add labels
            plt.xticks(angles[:-1], categories, size=12)
            
            # Draw y-axis labels (0 to 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], size=10)
            
            # Plot each model
            for i, row in radar_df.iterrows():
                model_name = row["Model"]
                values = row[radar_metrics].values.flatten().tolist()
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
                ax.fill(angles, values, alpha=0.1)
            
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            st.pyplot(fig)
            
            st.markdown("""
            In this radar chart:
            - Smaller values (closer to center) indicate better performance
            - All metrics are normalized relative to the maximum value
            - Better models have smaller radar footprints
            """)
    
    # Add correlation analysis if predictions are available
    moe_predictions = None
    if "MoE" in baseline_comparisons and isinstance(baseline_comparisons["MoE"], dict):
        moe_predictions = baseline_comparisons["MoE"].get("predictions", None)
    
    baseline_predictions = {}
    for model, metrics in baseline_comparisons.items():
        if model != "MoE" and isinstance(metrics, dict):
            predictions = metrics.get("predictions", None)
            if predictions is not None:
                baseline_predictions[model] = predictions
    
    if moe_predictions is not None and baseline_predictions:
        st.subheader("Error Correlation Analysis")
        
        for baseline, predictions in baseline_predictions.items():
            if len(predictions) == len(moe_predictions):
                # Calculate errors
                actual = baseline_comparisons["MoE"].get("actual", None)
                
                if actual is not None and len(actual) == len(moe_predictions):
                    moe_errors = np.array(moe_predictions) - np.array(actual)
                    baseline_errors = np.array(predictions) - np.array(actual)
                    
                    # Calculate correlation between errors
                    error_corr = np.corrcoef(moe_errors, baseline_errors)[0, 1]
                    
                    st.markdown(f"**MoE vs {baseline} Error Correlation: {error_corr:.4f}**")
                    
                    # Create scatter plot of errors
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    ax.scatter(baseline_errors, moe_errors, alpha=0.5)
                    ax.set_title(f"MoE vs {baseline} Error Comparison")
                    ax.set_xlabel(f"{baseline} Error")
                    ax.set_ylabel("MoE Error")
                    
                    # Add a line for equal errors
                    min_err = min(np.min(moe_errors), np.min(baseline_errors))
                    max_err = max(np.max(moe_errors), np.max(baseline_errors))
                    ax.plot([min_err, max_err], [min_err, max_err], 'r--', alpha=0.3)
                    
                    # Add lines at zero
                    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Calculate improvement statistics
                    moe_abs_errors = np.abs(moe_errors)
                    baseline_abs_errors = np.abs(baseline_errors)
                    
                    # Count where MoE is better
                    moe_better_count = np.sum(moe_abs_errors < baseline_abs_errors)
                    baseline_better_count = np.sum(baseline_abs_errors < moe_abs_errors)
                    equal_count = np.sum(moe_abs_errors == baseline_abs_errors)
                    
                    total_count = len(moe_errors)
                    moe_better_pct = moe_better_count / total_count * 100
                    baseline_better_pct = baseline_better_count / total_count * 100
                    equal_pct = equal_count / total_count * 100
                    
                    st.markdown(f"""
                    - MoE has lower error in **{moe_better_count}** cases ({moe_better_pct:.1f}%)
                    - {baseline} has lower error in **{baseline_better_count}** cases ({baseline_better_pct:.1f}%)
                    - Equal error in **{equal_count}** cases ({equal_pct:.1f}%)
                    """)
                    
                    # Add quadrant analysis
                    q1 = np.sum((moe_errors > 0) & (baseline_errors > 0))  # Both overestimate
                    q2 = np.sum((moe_errors < 0) & (baseline_errors > 0))  # MoE underestimates, baseline overestimates
                    q3 = np.sum((moe_errors < 0) & (baseline_errors < 0))  # Both underestimate
                    q4 = np.sum((moe_errors > 0) & (baseline_errors < 0))  # MoE overestimates, baseline underestimates
                    
                    st.markdown(f"""
                    **Error Direction Analysis:**
                    - Both models overestimate: **{q1}** cases ({q1/total_count*100:.1f}%)
                    - Both models underestimate: **{q3}** cases ({q3/total_count*100:.1f}%)
                    - MoE underestimates, {baseline} overestimates: **{q2}** cases ({q2/total_count*100:.1f}%)
                    - MoE overestimates, {baseline} underestimates: **{q4}** cases ({q4/total_count*100:.1f}%)
                    """)
