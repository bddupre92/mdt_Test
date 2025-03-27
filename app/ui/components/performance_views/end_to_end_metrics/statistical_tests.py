"""
Statistical Tests Component for End-to-End Performance Metrics

This module provides statistical analysis of model performance, including
hypothesis testing, confidence intervals, and significance testing for
performance comparisons between models.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import logging
import scipy.stats as stats

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

logger = logging.getLogger(__name__)

from app.ui.components.performance_views.helpers import safe_get_metric

def render_statistical_tests(end_to_end_metrics: Dict[str, Any]):
    """
    Render the statistical tests component for end-to-end performance metrics.
    
    Args:
        end_to_end_metrics: Dictionary containing end-to-end performance metrics
    """
    st.subheader("Statistical Tests & Analysis")
    
    # Extract statistical tests data
    statistical_tests = safe_get_metric(end_to_end_metrics, "statistical_tests", {})
    
    if not statistical_tests or not isinstance(statistical_tests, dict):
        st.info("No statistical tests data available")
        
        st.markdown("""
        ### Getting Started with Statistical Tests
        
        To generate statistical tests data:
        
        1. Run your model multiple times with different data splits or seeds
        2. Collect performance metrics for each run
        3. Use statistical testing functions to analyze the results
        
        Example code:
        ```python
        from baseline_comparison.moe_metrics import MoEMetricsCalculator
        import scipy.stats as stats
        
        # Run statistical significance tests
        rmse_values = [run1_rmse, run2_rmse, ..., runN_rmse]
        baseline_rmse_values = [baseline1_rmse, baseline2_rmse, ..., baselineN_rmse]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(rmse_values, baseline_rmse_values)
        
        # Update the system state
        system_state.performance_metrics["end_to_end_metrics"]["statistical_tests"] = {
            "t_test": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            },
            "performance_runs": {
                "moe": rmse_values,
                "baseline": baseline_rmse_values
            }
        }
        state_manager.save_state(system_state, "path/to/checkpoint")
        ```
        """)
        return
    
    # Create tabs for different statistical analyses
    stat_tabs = st.tabs([
        "Significance Tests",
        "Confidence Intervals",
        "Performance Distribution",
        "Hypothesis Testing"
    ])
    
    with stat_tabs[0]:
        render_significance_tests(statistical_tests)
    
    with stat_tabs[1]:
        render_confidence_intervals(statistical_tests)
    
    with stat_tabs[2]:
        render_performance_distribution(statistical_tests)
    
    with stat_tabs[3]:
        render_hypothesis_testing(statistical_tests)

def render_significance_tests(statistical_tests: Dict[str, Any]):
    """
    Render significance tests for model performance.
    
    Args:
        statistical_tests: Dictionary containing statistical tests data
    """
    st.subheader("Statistical Significance Tests")
    
    # Check for test results
    test_results = statistical_tests.get("test_results", {})
    
    if not test_results or not isinstance(test_results, dict):
        st.info("No significance test results available")
        return
    
    # Display test results in a table
    test_data = []
    
    for test_name, test_info in test_results.items():
        if isinstance(test_info, dict):
            test_row = {
                "Test": test_name,
                "Statistic": test_info.get("statistic", "N/A"),
                "p-value": test_info.get("p_value", "N/A"),
                "Significant": test_info.get("significant", False),
                "Description": test_info.get("description", "")
            }
            
            test_data.append(test_row)
    
    if not test_data:
        st.info("No valid test results found")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(test_data)
    
    # Format the table
    formatted_df = df.copy()
    for col in ["Statistic", "p-value"]:
        formatted_df[col] = formatted_df[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else x
        )
    
    # Add visual indicators for significance
    formatted_df["Significant"] = formatted_df["Significant"].apply(
        lambda x: "✅" if x else "❌"
    )
    
    # Reorder columns for better display
    formatted_df = formatted_df[["Test", "Statistic", "p-value", "Significant", "Description"]]
    
    st.dataframe(formatted_df, use_container_width=True)
    
    # Display a visual summary of significance
    significant_tests = sum(1 for test in test_data if test["Significant"])
    total_tests = len(test_data)
    
    if total_tests > 0:
        st.markdown(f"**{significant_tests} out of {total_tests} tests show statistical significance (p < 0.05)**")
        
        # Create a pie chart for visual summary
        fig, ax = plt.subplots(figsize=(6, 6))
        
        labels = ['Significant', 'Not Significant']
        sizes = [significant_tests, total_tests - significant_tests]
        colors = ['#4CAF50', '#F44336']
        
        ax.pie(
            sizes, 
            labels=labels, 
            colors=colors, 
            autopct='%1.1f%%', 
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        ax.axis('equal')
        
        st.pyplot(fig)

def render_confidence_intervals(statistical_tests: Dict[str, Any]):
    """
    Render confidence intervals for model performance metrics.
    
    Args:
        statistical_tests: Dictionary containing statistical tests data
    """
    st.subheader("Performance Confidence Intervals")
    
    # Check for confidence interval data
    ci_data = statistical_tests.get("confidence_intervals", {})
    
    if not ci_data or not isinstance(ci_data, dict):
        st.info("No confidence interval data available")
        return
    
    # Display confidence intervals for each metric
    ci_rows = []
    
    for metric, ci_info in ci_data.items():
        if isinstance(ci_info, dict):
            mean = ci_info.get("mean", None)
            lower = ci_info.get("lower_bound", None)
            upper = ci_info.get("upper_bound", None)
            std_dev = ci_info.get("std_dev", None)
            std_error = ci_info.get("std_error", None)
            confidence = ci_info.get("confidence", 0.95)
            
            if mean is not None and lower is not None and upper is not None:
                ci_rows.append({
                    "Metric": metric,
                    "Mean": mean,
                    "Lower Bound": lower,
                    "Upper Bound": upper,
                    "Std Dev": std_dev,
                    "Std Error": std_error,
                    "Confidence": confidence
                })
    
    if not ci_rows:
        st.info("No valid confidence interval data found")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(ci_rows)
    
    # Format the table
    formatted_df = df.copy()
    for col in ["Mean", "Lower Bound", "Upper Bound", "Std Dev", "Std Error"]:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
            )
    
    if "Confidence" in formatted_df.columns:
        formatted_df["Confidence"] = formatted_df["Confidence"].apply(
            lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
        )
    
    st.dataframe(formatted_df, use_container_width=True)
    
    # Create visualization of confidence intervals
    if not df.empty:
        # Allow user to select which metrics to visualize
        available_metrics = list(df["Metric"])
        
        selected_metrics = st.multiselect(
            "Select Metrics for Visualization",
            options=available_metrics,
            default=available_metrics[:3] if len(available_metrics) > 3 else available_metrics
        )
        
        if selected_metrics:
            filtered_df = df[df["Metric"].isin(selected_metrics)]
            
            if not filtered_df.empty:
                # Create the plot
                fig, ax = plt.subplots(figsize=(12, max(6, len(filtered_df) * 0.5)))
                
                # Extract data for plotting
                metrics = filtered_df["Metric"]
                means = filtered_df["Mean"]
                lowers = filtered_df["Lower Bound"]
                uppers = filtered_df["Upper Bound"]
                
                # Create error bars representing confidence intervals
                y_pos = np.arange(len(metrics))
                xerr = np.array([means - lowers, uppers - means])
                
                ax.errorbar(
                    means, y_pos, 
                    xerr=xerr,
                    fmt='o', 
                    capsize=5,
                    markersize=8,
                    linewidth=2,
                    elinewidth=2,
                    capthick=2
                )
                
                # Add labels for the mean values
                for i, mean in enumerate(means):
                    ax.text(
                        mean, 
                        y_pos[i] + 0.1, 
                        f'{mean:.4f}',
                        va='center',
                        fontweight='bold'
                    )
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(metrics)
                ax.set_xlabel('Value')
                ax.set_title('Confidence Intervals for Selected Metrics')
                
                # Add grid for readability
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display interpretation for a selected metric
                if selected_metrics:
                    selected_metric = st.selectbox(
                        "Select a Metric for Interpretation",
                        options=selected_metrics
                    )
                    
                    metric_data = filtered_df[filtered_df["Metric"] == selected_metric].iloc[0]
                    
                    mean = metric_data["Mean"]
                    lower = metric_data["Lower Bound"]
                    upper = metric_data["Upper Bound"]
                    confidence = metric_data["Confidence"]
                    
                    st.markdown(f"""
                    ### Interpretation for {selected_metric}
                    
                    With {confidence*100:.1f}% confidence, the true value of {selected_metric} is between **{lower:.4f}** and **{upper:.4f}**, with a point estimate of **{mean:.4f}**.
                    
                    This means:
                    - There is a {confidence*100:.1f}% probability that the interval [{lower:.4f}, {upper:.4f}] contains the true value.
                    - There is a {(1-confidence)*100:.1f}% probability that the true value falls outside this interval.
                    - The width of this interval ({upper-lower:.4f}) reflects the precision of our estimate.
                    """)
                    
                    # Add additional context based on the metric
                    metric_lower = selected_metric.lower()
                    if metric_lower in ["rmse", "mae", "mse", "error"]:
                        st.markdown("""
                        For error metrics like this one, **lower values are better**. A narrower confidence interval indicates more consistent performance across different runs or data splits.
                        """)
                    elif metric_lower in ["r2", "accuracy", "precision", "recall", "f1"]:
                        st.markdown("""
                        For this performance metric, **higher values are better**. A narrower confidence interval indicates more consistent performance across different runs or data splits.
                        """)

def render_performance_distribution(statistical_tests: Dict[str, Any]):
    """
    Render distribution of performance metrics across multiple runs.
    
    Args:
        statistical_tests: Dictionary containing statistical tests data
    """
    st.subheader("Performance Distribution Analysis")
    
    # Check for performance runs data
    performance_runs = statistical_tests.get("performance_runs", {})
    
    if not performance_runs or not isinstance(performance_runs, dict):
        st.info("No performance runs data available")
        return
    
    # Get available models and metrics
    models = []
    metrics = set()
    
    for model, model_data in performance_runs.items():
        if isinstance(model_data, dict):
            models.append(model)
            metrics.update(model_data.keys())
    
    if not models or not metrics:
        st.info("No valid performance runs data found")
        return
    
    # Allow user to select a metric for analysis
    metric_options = sorted(list(metrics))
    
    selected_metric = st.selectbox(
        "Select Metric for Distribution Analysis",
        options=metric_options,
        index=metric_options.index("rmse") if "rmse" in metric_options else 0
    )
    
    if not selected_metric:
        st.info("Please select a metric for analysis")
        return
    
    # Gather distribution data for the selected metric
    distribution_data = {}
    
    for model, model_data in performance_runs.items():
        if isinstance(model_data, dict) and selected_metric in model_data:
            metric_values = model_data[selected_metric]
            
            if isinstance(metric_values, list) and metric_values:
                distribution_data[model] = metric_values
    
    if not distribution_data:
        st.info(f"No distribution data available for {selected_metric}")
        return
    
    # Create distribution visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model, values in distribution_data.items():
        # Create kernel density estimate
        sns.kdeplot(values, label=model, ax=ax)
        
        # Add vertical line for mean
        mean_val = np.mean(values)
        ax.axvline(x=mean_val, linestyle='--', color=ax.lines[-1].get_color(), alpha=0.7)
        
        # Add label for mean
        ax.text(
            mean_val, 
            ax.get_ylim()[1] * 0.9,
            f'{model}: μ={mean_val:.4f}',
            rotation=90,
            color=ax.lines[-1].get_color(),
            ha='center',
            va='top'
        )
    
    ax.set_title(f"Distribution of {selected_metric.upper()} Across Multiple Runs")
    ax.set_xlabel(selected_metric.upper())
    ax.set_ylabel("Density")
    
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Create a box plot for comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for box plot
    box_data = []
    labels = []
    
    for model, values in distribution_data.items():
        box_data.append(values)
        labels.append(model)
    
    ax.boxplot(box_data, labels=labels, patch_artist=True)
    
    ax.set_title(f"Distribution of {selected_metric.upper()} by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel(selected_metric.upper())
    
    # Add grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display distribution statistics
    st.subheader("Distribution Statistics")
    
    stats_data = []
    
    for model, values in distribution_data.items():
        # Calculate statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        q1_val = np.percentile(values, 25)
        q3_val = np.percentile(values, 75)
        iqr_val = q3_val - q1_val
        
        stats_data.append({
            "Model": model,
            "Mean": mean_val,
            "Median": median_val,
            "Std Dev": std_val,
            "Min": min_val,
            "Max": max_val,
            "25th Percentile": q1_val,
            "75th Percentile": q3_val,
            "IQR": iqr_val,
            "CV (%)": (std_val / abs(mean_val)) * 100 if mean_val != 0 else np.nan
        })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        # Format for display
        formatted_stats = stats_df.copy()
        for col in stats_df.columns:
            if col != "Model":
                formatted_stats[col] = formatted_stats[col].apply(
                    lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"
                )
        
        st.dataframe(formatted_stats, use_container_width=True)
        
        # Perform distribution comparison tests if there are multiple models
        if len(distribution_data) > 1:
            st.subheader("Distribution Comparison Tests")
            
            comparison_results = []
            
            # Get all pairs of models
            models = list(distribution_data.keys())
            
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    model1 = models[i]
                    model2 = models[j]
                    
                    values1 = distribution_data[model1]
                    values2 = distribution_data[model2]
                    
                    # Perform t-test
                    t_stat, t_p = stats.ttest_ind(values1, values2, equal_var=False)
                    
                    # Perform Mann-Whitney U test
                    u_stat, u_p = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                    
                    # Perform Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.ks_2samp(values1, values2)
                    
                    comparison_results.append({
                        "Model 1": model1,
                        "Model 2": model2,
                        "t-Test p-value": t_p,
                        "t-Test Significant": t_p < 0.05,
                        "Mann-Whitney p-value": u_p,
                        "Mann-Whitney Significant": u_p < 0.05,
                        "KS Test p-value": ks_p,
                        "KS Test Significant": ks_p < 0.05
                    })
            
            if comparison_results:
                # Convert to DataFrame
                comp_df = pd.DataFrame(comparison_results)
                
                # Format p-values
                for col in ["t-Test p-value", "Mann-Whitney p-value", "KS Test p-value"]:
                    comp_df[col] = comp_df[col].apply(lambda x: f"{x:.4f}")
                
                # Format significance indicators
                for col in ["t-Test Significant", "Mann-Whitney Significant", "KS Test Significant"]:
                    comp_df[col] = comp_df[col].apply(lambda x: "✅" if x else "❌")
                
                st.dataframe(comp_df, use_container_width=True)
                
                st.markdown("""
                **Test Interpretations:**
                - **t-Test**: Tests if the means of the two distributions are significantly different
                - **Mann-Whitney U Test**: Non-parametric test that compares the medians of two distributions
                - **Kolmogorov-Smirnov Test**: Tests if two samples come from the same distribution
                
                ✅ = Significant difference (p < 0.05)
                ❌ = No significant difference (p ≥ 0.05)
                """)

def render_hypothesis_testing(statistical_tests: Dict[str, Any]):
    """
    Render hypothesis testing results.
    
    Args:
        statistical_tests: Dictionary containing statistical tests data
    """
    st.subheader("Hypothesis Testing Results")
    
    # Check for hypothesis testing data
    hypothesis_tests = statistical_tests.get("hypothesis_tests", {})
    
    if not hypothesis_tests or not isinstance(hypothesis_tests, dict):
        st.info("No hypothesis testing data available")
        
        st.markdown("""
        ### Hypothesis Testing
        
        Hypothesis testing allows you to make statistical inferences about your model performance.
        Common hypotheses to test include:
        
        - Is the MoE model significantly better than baseline models?
        - Is there a significant difference between expert models?
        - Does feature X significantly impact model performance?
        
        To generate hypothesis testing data, formulate null and alternative hypotheses,
        collect performance data, and use statistical tests to evaluate the hypotheses.
        """)
        return
    
    # Display hypothesis test results
    for test_name, test_data in hypothesis_tests.items():
        if isinstance(test_data, dict):
            null_hypothesis = test_data.get("null_hypothesis", "No null hypothesis specified")
            alternative_hypothesis = test_data.get("alternative_hypothesis", "No alternative hypothesis specified")
            p_value = test_data.get("p_value", None)
            test_statistic = test_data.get("test_statistic", None)
            significant = test_data.get("significant", False)
            alpha = test_data.get("alpha", 0.05)
            conclusion = test_data.get("conclusion", "No conclusion provided")
            
            # Format the result with appropriate styling
            if significant:
                result_style = "background-color: #d4edda; padding: 10px; border-radius: 5px;"
                result_text = "REJECT null hypothesis"
            else:
                result_style = "background-color: #f8d7da; padding: 10px; border-radius: 5px;"
                result_text = "FAIL TO REJECT null hypothesis"
            
            # Create expandable section for each hypothesis test
            with st.expander(f"Hypothesis: {test_name}", expanded=True):
                st.markdown(f"""
                **Null Hypothesis (H₀):** {null_hypothesis}
                
                **Alternative Hypothesis (H₁):** {alternative_hypothesis}
                
                **Test Results:**
                - Test Statistic: {test_statistic if test_statistic is not None else 'N/A'}
                - p-value: {p_value:.4f if p_value is not None else 'N/A'}
                - Significance Level (α): {alpha:.3f}
                
                <div style="{result_style}">
                <strong>Conclusion:</strong> {result_text} (p{'<' if significant else '≥'}α)
                </div>
                
                **Interpretation:** {conclusion}
                """, unsafe_allow_html=True)
                
                # Visualize the p-value against alpha
                if p_value is not None:
                    fig, ax = plt.subplots(figsize=(10, 3))
                    
                    # Create a horizontal line representing p-values from 0 to 1
                    ax.plot([0, 1], [0, 0], 'k-', linewidth=2)
                    
                    # Mark the critical value (alpha)
                    ax.axvline(x=alpha, color='r', linestyle='--', label=f'α = {alpha}')
                    
                    # Mark the p-value
                    ax.plot(p_value, 0, 'bo', markersize=10, label=f'p = {p_value:.4f}')
                    
                    # Set labels and title
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.1, 0.1)
                    ax.set_xlabel('p-value')
                    ax.set_title('Significance Test Visualization')
                    
                    # Remove y-axis ticks and labels
                    ax.set_yticks([])
                    
                    # Shade rejection region
                    ax.fill_between([0, alpha], -0.1, 0.1, color='red', alpha=0.2, label='Rejection Region')
                    
                    # Add rejection and non-rejection region labels
                    ax.text(alpha/2, 0.05, 'Reject H₀', ha='center')
                    ax.text((1+alpha)/2, 0.05, 'Fail to Reject H₀', ha='center')
                    
                    ax.legend(loc='lower right')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    # Display power analysis if available
    power_analysis = statistical_tests.get("power_analysis", {})
    
    if power_analysis and isinstance(power_analysis, dict):
        st.subheader("Statistical Power Analysis")
        
        for test_name, power_data in power_analysis.items():
            if isinstance(power_data, dict):
                power = power_data.get("power", None)
                sample_size = power_data.get("sample_size", None)
                effect_size = power_data.get("effect_size", None)
                alpha = power_data.get("alpha", 0.05)
                
                if power is not None:
                    # Create expandable section for power analysis
                    with st.expander(f"Power Analysis: {test_name}", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Statistical Power",
                                f"{power:.2f}" if power is not None else "N/A",
                                help="Probability of correctly rejecting null hypothesis when it is false"
                            )
                        
                        with col2:
                            st.metric(
                                "Sample Size",
                                f"{sample_size}" if sample_size is not None else "N/A",
                                help="Number of observations used in the test"
                            )
                        
                        with col3:
                            st.metric(
                                "Effect Size",
                                f"{effect_size:.2f}" if effect_size is not None else "N/A",
                                help="Magnitude of the difference between groups"
                            )
                        
                        with col4:
                            st.metric(
                                "Significance Level",
                                f"{alpha:.2f}" if alpha is not None else "N/A",
                                help="Probability of Type I error (false positive)"
                            )
                        
                        # Add interpretation of power
                        if power is not None:
                            if power >= 0.8:
                                st.success(f"Power of {power:.2f} is adequate (≥0.80): The test has sufficient ability to detect the effect if it exists.")
                            elif power >= 0.5:
                                st.warning(f"Power of {power:.2f} is moderate: The test may fail to detect an effect even if it exists.")
                            else:
                                st.error(f"Power of {power:.2f} is low: The test is unlikely to detect an effect even if it exists.")
                        
                        # Add sample size recommendation if power is low
                        if power is not None and power < 0.8 and "recommended_sample_size" in power_data:
                            recommended_sample_size = power_data["recommended_sample_size"]
                            st.info(f"To achieve a power of 0.8, a sample size of approximately {recommended_sample_size} is recommended.")
                        
                        # Create power curve if all necessary data is available
                        if "power_curve" in power_data and isinstance(power_data["power_curve"], dict):
                            power_curve = power_data["power_curve"]
                            
                            if "sample_sizes" in power_curve and "powers" in power_curve:
                                sample_sizes = power_curve["sample_sizes"]
                                powers = power_curve["powers"]
                                
                                if len(sample_sizes) == len(powers):
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    ax.plot(sample_sizes, powers, 'b-', linewidth=2)
                                    
                                    # Mark the current sample size and power
                                    if sample_size is not None and power is not None:
                                        ax.plot(sample_size, power, 'ro', markersize=8)
                                        ax.text(sample_size, power, f' ({sample_size}, {power:.2f})', va='center')
                                    
                                    # Add a horizontal line at power = 0.8
                                    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.7)
                                    ax.text(min(sample_sizes), 0.82, 'Adequate Power (0.8)', va='center')
                                    
                                    ax.set_xlabel('Sample Size')
                                    ax.set_ylabel('Statistical Power')
                                    ax.set_title('Power Curve Analysis')
                                    
                                    ax.grid(True, linestyle='--', alpha=0.7)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
