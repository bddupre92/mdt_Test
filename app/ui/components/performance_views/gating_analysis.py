"""
Gating Network Analysis View for Performance Analysis

This module provides UI components for analyzing the performance of the gating network
within a Mixture of Experts (MoE) system, focusing on expert selection performance,
routing decisions, and optimality of weight distributions.
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

def render_gating_analysis(gating_evaluation: Dict[str, Any]):
    """
    Render the Gating Network Analysis view for detailed evaluation of the gating network.
    
    Args:
        gating_evaluation: Dictionary containing gating network evaluation metrics
    """
    st.header("Gating Network Analysis")
    
    st.markdown("""
    This view analyzes the performance of the gating network, which is responsible for 
    routing inputs to the appropriate expert models in the MoE system. A well-functioning
    gating network is critical for optimal MoE performance.
    """)
    
    if not gating_evaluation:
        st.warning("No gating network evaluation data available")
        
        st.markdown("""
        ### Getting Started with Gating Network Analysis
        
        To generate gating network metrics:
        
        1. Use the MoEMetricsCalculator to evaluate the gating network's performance
        2. The system will analyze expert selection patterns and optimality
        3. Results will be saved to the system state
        
        Example code:
        ```python
        from baseline_comparison.moe_metrics import MoEMetricsCalculator
        
        calculator = MoEMetricsCalculator()
        metrics = calculator.compute_all_metrics(
            y_true=y_test,
            y_pred=predictions,
            expert_weights=expert_weights,
            expert_outputs=individual_expert_outputs
        )
        
        # Update the system state
        system_state.performance_metrics["gating_evaluation"] = metrics["gating_evaluation"]
        state_manager.save_state(system_state, "path/to/checkpoint")
        ```
        """)
        return
    
    # Create tabs for different analysis views
    gating_tabs = st.tabs([
        "Selection Performance", 
        "Weight Distribution", 
        "Regret Analysis",
        "Expert Routing"
    ])
    
    with gating_tabs[0]:
        render_selection_performance(gating_evaluation)
    
    with gating_tabs[1]:
        render_weight_distribution(gating_evaluation)
    
    with gating_tabs[2]:
        render_regret_analysis(gating_evaluation)
    
    with gating_tabs[3]:
        render_expert_routing(gating_evaluation)

def safe_get_metric(data, key, default=None):
    """
    Safely access a metric regardless of whether it's in a dictionary or object.
    
    Args:
        data: Dictionary or object containing metrics
        key: The key or attribute name to access
        default: Default value to return if the key is not found
        
    Returns:
        The value associated with the key or the default value
    """
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)

def render_selection_performance(gating_evaluation: Dict[str, Any]):
    """
    Render gating network selection performance view.
    
    Args:
        gating_evaluation: Dictionary containing gating network evaluation metrics
    """
    st.subheader("Expert Selection Performance")
    
    # Extract selection metrics
    optimal_selection_rate = safe_get_metric(gating_evaluation, "optimal_expert_selection_rate")
    
    if optimal_selection_rate is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a gauge-like visualization for optimal selection rate
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Create a circle with a percentage fill
            wedgeprops = {'width': 0.3, 'edgecolor': 'white', 'linewidth': 2}
            ax.pie(
                [optimal_selection_rate, 1-optimal_selection_rate], 
                startangle=90, 
                counterclock=False,
                colors=['#1E88E5', '#ECEFF1'],
                wedgeprops=wedgeprops,
                labels=None
            )
            
            # Add percentage text
            ax.text(0, 0, f"{optimal_selection_rate:.1%}", 
                   ha='center', va='center', fontsize=24)
            
            ax.text(0, -0.15, "Optimal Expert\nSelection Rate", 
                   ha='center', va='center', fontsize=12)
            
            # Make the plot circular
            ax.set_aspect('equal')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            ### Optimal Selection Rate
            
            This metric shows how often the gating network selects the best possible expert 
            for a given input. A higher rate indicates better gating network performance.
            
            **What it means:**
            - **100%**: Perfect expert selection for all inputs
            - **>80%**: Excellent gating performance
            - **50-80%**: Good gating performance, some room for improvement
            - **<50%**: Poor gating decisions, significant improvement needed
            
            **How to improve:**
            - Re-train the gating network with more diverse data
            - Add more features to help the gating network differentiate between data types
            - Consider using a more complex gating architecture
            """)
    else:
        st.info("Optimal expert selection rate data not available")
    
    # Display selection confusion matrix if available
    if "expert_selection_confusion" in gating_evaluation:
        st.subheader("Expert Selection Confusion Matrix")
        
        confusion = gating_evaluation["expert_selection_confusion"]
        if isinstance(confusion, dict) and "matrix" in confusion and "expert_names" in confusion:
            matrix = confusion["matrix"]
            expert_names = confusion["expert_names"]
            
            # Create a heatmap of the confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(
                matrix, 
                annot=True, 
                fmt=".2f", 
                cmap="Blues",
                xticklabels=expert_names,
                yticklabels=expert_names,
                ax=ax
            )
            
            ax.set_title("Expert Selection Confusion Matrix")
            ax.set_xlabel("Selected Expert")
            ax.set_ylabel("Optimal Expert")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            The confusion matrix shows how often the gating network selected each expert (columns) 
            compared to which expert would have been optimal (rows). The diagonal represents 
            correct selections, while off-diagonal values represent sub-optimal selections.
            """)
        else:
            st.info("Expert selection confusion matrix is not in the expected format")

def render_weight_distribution(gating_evaluation: Dict[str, Any]):
    """
    Render gating network weight distribution analysis.
    
    Args:
        gating_evaluation: Dictionary containing gating network evaluation metrics
    """
    st.subheader("Gating Weight Distribution")
    
    # Extract weight distribution metrics
    weight_concentration = safe_get_metric(gating_evaluation, "weight_concentration")
    expert_usage = safe_get_metric(gating_evaluation, "expert_usage_distribution")
    
    if weight_concentration is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Weight Concentration", 
                f"{weight_concentration:.4f}",
                help="Measures how concentrated expert weights are. Higher values indicate more focused weight distribution."
            )
        
        with col2:
            shannon_entropy = safe_get_metric(gating_evaluation, "weight_shannon_entropy")
            if shannon_entropy is not None:
                st.metric(
                    "Weight Shannon Entropy", 
                    f"{shannon_entropy:.4f}",
                    help="Measures the diversity of weight distribution. Higher values indicate more evenly distributed weights."
                )
        
        with col3:
            gini_coefficient = safe_get_metric(gating_evaluation, "weight_gini_coefficient")
            if gini_coefficient is not None:
                st.metric(
                    "Weight Gini Coefficient", 
                    f"{gini_coefficient:.4f}",
                    help="Measures inequality in the weight distribution. Higher values indicate more unequal distribution of weights."
                )
    
    # Visualize expert usage distribution if available
    if expert_usage and isinstance(expert_usage, dict):
        st.subheader("Expert Usage Distribution")
        
        # Convert usage to DataFrame
        usage_data = []
        for expert_name, usage_percent in expert_usage.items():
            usage_data.append({
                "Expert": expert_name,
                "Usage (%)": usage_percent * 100
            })
        
        usage_df = pd.DataFrame(usage_data)
        
        # Sort by usage
        usage_df = usage_df.sort_values("Usage (%)", ascending=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, min(8, max(4, len(usage_df) * 0.5))))
        
        bars = ax.barh(usage_df["Expert"], usage_df["Usage (%)"])
        ax.set_xlabel("Usage (%)")
        ax.set_title("Expert Usage Distribution")
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 1, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%',
                ha='left', va='center'
            )
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add interpretation
        if len(usage_df) > 1:
            usage_min = usage_df["Usage (%)"].min()
            usage_max = usage_df["Usage (%)"].max()
            usage_ratio = usage_max / max(usage_min, 0.01)  # Avoid division by zero
            
            if usage_ratio > 10:
                st.warning(
                    f"The usage distribution is highly skewed (max/min ratio: {usage_ratio:.1f}x). "
                    "Consider rebalancing your expert models or training data to ensure more even utilization."
                )
            elif usage_ratio > 3:
                st.info(
                    f"The usage distribution shows moderate skew (max/min ratio: {usage_ratio:.1f}x). "
                    "This may be expected if your data naturally clusters into different sizes of groups."
                )
            else:
                st.success(
                    f"The usage distribution is fairly balanced (max/min ratio: {usage_ratio:.1f}x). "
                    "Your gating network is effectively utilizing the different expert models."
                )
    
    # Visualize weight distribution over samples if available
    weight_distributions = safe_get_metric(gating_evaluation, "weight_distributions")
    if weight_distributions is not None and isinstance(weight_distributions, dict):
        st.subheader("Expert Weight Distribution Over Samples")
        
        # Get sorted expert names based on average weight
        expert_names = []
        avg_weights = []
        
        # Create matrix from weight distributions
        if "samples" in weight_distributions and "expert_names" in weight_distributions:
            sample_weights = weight_distributions["samples"]
            expert_names = weight_distributions["expert_names"]
            
            if isinstance(sample_weights, list) and sample_weights and isinstance(expert_names, list):
                # Convert to numpy array for easier handling
                weight_matrix = np.array(sample_weights)
                
                # Calculate average weights for sorting
                avg_weights = np.mean(weight_matrix, axis=0)
                
                # Limit number of samples to show (for performance)
                max_samples = min(300, len(weight_matrix))
                if len(weight_matrix) > max_samples:
                    st.info(f"Showing weight distribution for {max_samples} random samples out of {len(weight_matrix)} total samples")
                    # Take a random subset
                    indices = np.random.choice(len(weight_matrix), max_samples, replace=False)
                    weight_matrix = weight_matrix[indices]
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Sort by average weights if there are more than 1 expert
                if len(expert_names) > 1:
                    sorted_indices = np.argsort(-avg_weights)  # Descending order
                    weight_matrix = weight_matrix[:, sorted_indices]
                    sorted_expert_names = [expert_names[i] for i in sorted_indices]
                else:
                    sorted_expert_names = expert_names
                
                sns.heatmap(
                    weight_matrix, 
                    cmap="viridis",
                    xticklabels=sorted_expert_names,
                    yticklabels=False,
                    ax=ax
                )
                
                ax.set_title("Expert Weight Distribution Across Samples")
                ax.set_xlabel("Expert")
                ax.set_ylabel("Sample")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show weight statistics
                st.subheader("Expert Weight Statistics")
                
                stats_data = []
                for i, expert in enumerate(expert_names):
                    weights = weight_matrix[:, i] if i < weight_matrix.shape[1] else []
                    
                    if len(weights) > 0:
                        stats_data.append({
                            "Expert": expert,
                            "Mean Weight": np.mean(weights),
                            "Median Weight": np.median(weights),
                            "Max Weight": np.max(weights),
                            "Min Weight": np.min(weights),
                            "Std Dev": np.std(weights),
                            "Dominance %": np.mean(weights > 0.5) * 100  # % of samples where weight > 0.5
                        })
                
                stats_df = pd.DataFrame(stats_data)
                
                # Format for display
                formatted_df = stats_df.copy()
                for col in ["Mean Weight", "Median Weight", "Max Weight", "Min Weight", "Std Dev"]:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
                
                formatted_df["Dominance %"] = formatted_df["Dominance %"].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(formatted_df)
            else:
                st.info("Weight distribution data is not in the expected format")
        else:
            st.info("Weight distribution data is not in the expected format")

def render_regret_analysis(gating_evaluation: Dict[str, Any]):
    """
    Render gating network regret analysis.
    
    Args:
        gating_evaluation: Dictionary containing gating network evaluation metrics
    """
    st.subheader("Expert Selection Regret Analysis")
    
    st.markdown("""
    **Regret** in a Mixture of Experts system refers to the performance loss from selecting 
    a sub-optimal expert for a given input. It's calculated as the difference between the error
    of the selected expert and the error of the best possible expert for each input.
    """)
    
    # Extract regret metrics
    mean_regret = safe_get_metric(gating_evaluation, "mean_regret")
    max_regret = safe_get_metric(gating_evaluation, "max_regret")
    regret_distribution = safe_get_metric(gating_evaluation, "regret_distribution")
    
    if mean_regret is not None or max_regret is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if mean_regret is not None:
                st.metric(
                    "Mean Regret", 
                    f"{mean_regret:.4f}",
                    help="Average performance loss due to sub-optimal expert selection. Lower is better."
                )
        
        with col2:
            if max_regret is not None:
                st.metric(
                    "Max Regret", 
                    f"{max_regret:.4f}",
                    help="Worst-case performance loss due to sub-optimal expert selection. Lower is better."
                )
    
    # Visualize regret distribution if available
    if regret_distribution is not None and isinstance(regret_distribution, dict):
        if "bins" in regret_distribution and "counts" in regret_distribution:
            bins = regret_distribution["bins"]
            counts = regret_distribution["counts"]
            
            if isinstance(bins, list) and isinstance(counts, list) and len(bins) > 1 and len(counts) > 0:
                # Create histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot distribution
                ax.bar(bins[:-1], counts, width=bins[1]-bins[0], align="edge", alpha=0.7)
                ax.set_title("Regret Distribution")
                ax.set_xlabel("Regret Value")
                ax.set_ylabel("Count")
                
                # Add vertical line for mean regret
                if mean_regret is not None:
                    ax.axvline(x=mean_regret, color='r', linestyle='--', label=f"Mean: {mean_regret:.4f}")
                    ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Calculate percentage of low-regret decisions
                if len(bins) > 1 and len(counts) > 0:
                    low_regret_threshold = 0.05  # Arbitrary threshold for "low regret"
                    
                    # Find bin index for the threshold
                    threshold_bin_idx = np.searchsorted(bins, low_regret_threshold) - 1
                    threshold_bin_idx = max(0, min(threshold_bin_idx, len(counts) - 1))
                    
                    # Calculate sum of counts for bins below threshold
                    low_regret_count = sum(counts[:threshold_bin_idx + 1])
                    total_count = sum(counts)
                    
                    if total_count > 0:
                        low_regret_percentage = (low_regret_count / total_count) * 100
                        
                        st.markdown(f"""
                        **Low Regret Decisions**: {low_regret_percentage:.1f}% of decisions had a regret below {low_regret_threshold:.2f}
                        
                        This means that for {low_regret_percentage:.1f}% of inputs, the gating network's expert selection was
                        very close to optimal (within {low_regret_threshold:.2f} error units of the best possible expert).
                        """)
            else:
                st.info("Regret distribution data is not in the expected format")
        else:
            st.info("Regret distribution data is not in the expected format")
    
    # Add regret analysis over features if available
    feature_regret = safe_get_metric(gating_evaluation, "feature_regret")
    if feature_regret is not None and isinstance(feature_regret, dict):
        st.subheader("Regret Analysis by Feature")
        
        # Convert to DataFrame
        feature_regret_data = []
        for feature, regret in feature_regret.items():
            feature_regret_data.append({
                "Feature": feature,
                "Mean Regret": regret
            })
        
        regret_df = pd.DataFrame(feature_regret_data)
        
        # Sort by regret
        regret_df = regret_df.sort_values("Mean Regret", ascending=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, min(8, max(4, len(regret_df) * 0.5))))
        
        bars = ax.barh(regret_df["Feature"], regret_df["Mean Regret"])
        ax.set_xlabel("Mean Regret")
        ax.set_title("Regret by Feature")
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center'
            )
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        This chart shows which features are associated with the highest regret. Features
        with high regret may be areas where the gating network struggles to make optimal
        expert selections.
        
        **Potential actions:**
        - Focus on improving the gating network's performance for high-regret features
        - Consider adding more training data that focuses on these feature areas
        - Potentially create specialized experts for difficult feature regions
        """)

def render_expert_routing(gating_evaluation: Dict[str, Any]):
    """
    Render expert routing analysis.
    
    Args:
        gating_evaluation: Dictionary containing gating network evaluation metrics
    """
    st.subheader("Expert Routing Analysis")
    
    # Extract expert transition data
    expert_transitions = safe_get_metric(gating_evaluation, "expert_transitions")
    
    if expert_transitions is not None and isinstance(expert_transitions, dict):
        if "matrix" in expert_transitions and "expert_names" in expert_transitions:
            transition_matrix = expert_transitions["matrix"]
            expert_names = expert_transitions["expert_names"]
            
            if isinstance(transition_matrix, list) and isinstance(expert_names, list):
                # Convert to numpy if necessary
                if isinstance(transition_matrix, list):
                    transition_matrix = np.array(transition_matrix)
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                
                sns.heatmap(
                    transition_matrix, 
                    annot=True, 
                    fmt=".2f", 
                    cmap="Blues",
                    xticklabels=expert_names,
                    yticklabels=expert_names,
                    ax=ax
                )
                
                ax.set_title("Expert Transition Matrix")
                ax.set_xlabel("Next Expert")
                ax.set_ylabel("Current Expert")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                The expert transition matrix shows how frequently the gating network switches
                from one expert to another in sequential inputs. The diagonal represents
                staying with the same expert, while off-diagonal values represent transitions.
                
                **Interpretation:**
                - High diagonal values indicate stability in expert selection
                - High off-diagonal values indicate frequent expert switching
                - Balanced transitions suggest good specialization
                - Highly imbalanced transitions may indicate routing inefficiencies
                """)
                
                # Calculate and display transition metrics
                if transition_matrix.shape[0] > 1:
                    # Calculate stability (average of diagonal elements)
                    stability = np.trace(transition_matrix) / transition_matrix.shape[0]
                    
                    # Calculate entropy of transitions
                    entropy = 0
                    for i in range(transition_matrix.shape[0]):
                        row = transition_matrix[i]
                        row_sum = np.sum(row)
                        if row_sum > 0:
                            row_probs = row / row_sum
                            for p in row_probs:
                                if p > 0:
                                    entropy -= p * np.log2(p)
                    
                    # Normalize entropy by maximum possible entropy
                    max_entropy = np.log2(transition_matrix.shape[0])
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Expert Stability", 
                            f"{stability:.2f}",
                            help="Measures how often the same expert is used for consecutive inputs. Higher values indicate more stable expert selection."
                        )
                    
                    with col2:
                        st.metric(
                            "Transition Entropy", 
                            f"{normalized_entropy:.2f}",
                            help="Measures the diversity and unpredictability of expert transitions. Higher values indicate more diverse transitions."
                        )
                    
                    # Provide interpretation
                    if stability > 0.8:
                        st.success(
                            f"Expert stability is high ({stability:.2f}), indicating consistent expert selection across sequential inputs. "
                            "This suggests good differentiation between expert domains."
                        )
                    elif stability < 0.4:
                        st.warning(
                            f"Expert stability is low ({stability:.2f}), indicating frequent switching between experts. "
                            "This may suggest insufficient expert specialization or noisy input data."
                        )
                    else:
                        st.info(
                            f"Expert stability is moderate ({stability:.2f}), showing a balance between consistent expert selection and flexibility."
                        )
            else:
                st.info("Expert transition data is not in the expected format")
        else:
            st.info("Expert transition data is not in the expected format")
    
    # Visualize feature-based routing if available
    feature_routing = safe_get_metric(gating_evaluation, "feature_based_routing")
    if feature_routing is not None and isinstance(feature_routing, dict):
        st.subheader("Feature-Based Routing Analysis")
        
        feature_names = feature_routing.get("feature_names", [])
        expert_names = feature_routing.get("expert_names", [])
        importance_matrix = feature_routing.get("importance_matrix")
        
        if isinstance(feature_names, list) and isinstance(expert_names, list) and importance_matrix is not None:
            # Convert to numpy if necessary
            if isinstance(importance_matrix, list):
                importance_matrix = np.array(importance_matrix)
            
            # Create heatmap if there's enough data
            if len(feature_names) > 0 and len(expert_names) > 0 and importance_matrix.shape == (len(expert_names), len(feature_names)):
                fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 0.5), max(6, len(expert_names) * 0.5)))
                
                sns.heatmap(
                    importance_matrix, 
                    cmap="viridis",
                    xticklabels=feature_names,
                    yticklabels=expert_names,
                    ax=ax
                )
                
                ax.set_title("Feature Importance for Expert Selection")
                ax.set_xlabel("Feature")
                ax.set_ylabel("Expert")
                
                # Rotate x-axis labels for better readability if many features
                plt.xticks(rotation=45, ha="right")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                This heatmap shows the importance of each feature in routing inputs to different experts.
                Brighter colors indicate higher importance of a feature for routing to a particular expert.
                
                **Key insights:**
                - Features with distinct bright areas across experts are strong routing signals
                - Features with uniform coloring may have limited routing influence
                - Identifying the most influential features helps understand gating network decisions
                """)
                
                # Show top routing features for each expert
                st.subheader("Top Routing Features by Expert")
                
                for i, expert in enumerate(expert_names):
                    if i < importance_matrix.shape[0]:
                        importances = importance_matrix[i]
                        
                        # Get indices of top features
                        top_n = min(5, len(feature_names))
                        top_indices = np.argsort(-importances)[:top_n]
                        
                        # Create list of top features with importance scores
                        top_features = [
                            f"**{feature_names[idx]}** ({importances[idx]:.4f})"
                            for idx in top_indices
                        ]
                        
                        st.markdown(f"**{expert}**: " + ", ".join(top_features))
            else:
                st.info("Feature routing data dimensions don't match feature and expert names")
        else:
            st.info("Feature routing data is not in the expected format")
    
    # Add routing recommendations if available
    routing_recommendations = safe_get_metric(gating_evaluation, "routing_recommendations")
    if routing_recommendations is not None and isinstance(routing_recommendations, list):
        st.subheader("Routing Improvement Recommendations")
        
        for i, recommendation in enumerate(routing_recommendations):
            if isinstance(recommendation, dict) and "description" in recommendation:
                st.markdown(f"**{i+1}. {recommendation['description']}**")
                
                # Show additional details if available
                if "rationale" in recommendation:
                    st.markdown(recommendation["rationale"])
                
                if "impact" in recommendation:
                    st.info(f"Expected impact: {recommendation['impact']}")
                
                st.divider()
            elif isinstance(recommendation, str):
                st.markdown(f"**{i+1}.** {recommendation}")
                st.divider()
