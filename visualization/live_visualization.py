"""
Real-time visualization tools for MOE Framework.

This module provides real-time visualization capabilities for monitoring
training progress, optimization algorithms, and convergence of models.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

class LiveOptimizationMonitor:
    """
    Class for monitoring optimization processes in real-time.
    Used by the meta-optimizer component to visualize optimization progress.
    """
    def __init__(self, title="Optimization Monitor"):
        """
        Initialize the optimization monitor.
        
        Args:
            title (str): Title for the monitor display
        """
        self.title = title
        self.trials = []
        self.best_score = 0
        self.best_params = {}
        self.best_trial = 0
        self.iteration = 0
        logger.info(f"Initialized {self.title}")
    
    def update(self, trial_result: Dict[str, Any]):
        """
        Update the monitor with a new trial result.
        
        Args:
            trial_result: Dictionary containing trial information
                Required keys:
                - score: The evaluation score for this trial
                - params: Dictionary of parameter values used in this trial
        """
        self.iteration += 1
        score = trial_result.get('score', 0)
        params = trial_result.get('params', {})
        
        # Check if this is the best score
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            self.best_trial = self.iteration
        
        # Add trial data
        trial_data = {
            "trial_id": self.iteration,
            "score": score,
        }
        
        # Add parameter data
        for param_name, param_value in params.items():
            trial_data[f"param_{param_name}"] = param_value
        
        self.trials.append(trial_data)
        logger.info(f"Updated monitor with trial {self.iteration}, score: {score:.4f}")
        
        return {
            "best_score": self.best_score,
            "best_params": self.best_params,
            "best_trial": self.best_trial
        }
    
    def reset(self):
        """Reset the monitor to its initial state."""
        self.trials = []
        self.best_score = 0
        self.best_params = {}
        self.best_trial = 0
        self.iteration = 0
        logger.info(f"Reset {self.title}")
    
    def display(self, container=None):
        """
        Display the optimization monitor in a Streamlit container.
        
        Args:
            container: Optional Streamlit container to render in
        """
        # If no container is provided, use st directly
        if container is None:
            container = st
        
        container.markdown(f"## {self.title}")
        
        if not self.trials:
            container.info("No optimization trials completed yet.")
            return
        
        # Create DataFrame from trials
        trials_df = pd.DataFrame(self.trials)
        
        # Display trials visualization
        fig = px.scatter(
            trials_df,
            x="trial_id",
            y="score",
            color="score",
            size="score",
            color_continuous_scale="Viridis",
            title="Trial Scores"
        )
        
        # Add best score line
        fig.add_hline(
            y=self.best_score,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Best: {self.best_score:.4f}"
        )
        
        container.plotly_chart(fig, use_container_width=True)
        
        # Display best configuration
        container.markdown("### Best Configuration")
        container.markdown(f"**Best Score:** {self.best_score:.4f}")
        container.markdown(f"**Found at Trial:** {self.best_trial}")
        
        container.markdown("**Best Parameters:**")
        for param, value in self.best_params.items():
            container.markdown(f"- {param}: {value}")
        
        # Check if we have enough trials for parameter importance
        if len(self.trials) >= 5:
            # Create parallel coordinates plot for parameters
            param_cols = [col for col in trials_df.columns if col.startswith("param_")]
            
            if param_cols:
                # Display parameter importance
                container.markdown("### Parameter Exploration")
                fig = px.parallel_coordinates(
                    trials_df,
                    dimensions=param_cols + ["score"],
                    color="score",
                    color_continuous_scale="Viridis",
                    title="Parameter Space Exploration"
                )
                
                container.plotly_chart(fig, use_container_width=True)
                
                # Calculate parameter importance (simplified)
                container.markdown("### Parameter Importance")
                importance = {}
                
                for param in param_cols:
                    # Calculate correlation with score as a simple importance metric
                    correlation = abs(trials_df[param].corr(trials_df["score"]))
                    importance[param.replace("param_", "")] = correlation
                
                # Create importance chart
                importance_df = pd.DataFrame({
                    "Parameter": list(importance.keys()),
                    "Importance": list(importance.values())
                }).sort_values("Importance", ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x="Parameter",
                    y="Importance",
                    color="Importance",
                    color_continuous_scale="Viridis",
                    title="Parameter Importance (Correlation with Score)"
                )
                
                container.plotly_chart(fig, use_container_width=True)

def create_live_training_monitor():
    """
    Creates a real-time training monitor visualization that updates as training progresses.
    """
    st.markdown("## Live Training Monitor")
    
    # Create placeholders for the visualizations
    progress_placeholder = st.empty()
    loss_chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Initialize session state for training data if not already done
    if 'training_progress' not in st.session_state:
        st.session_state.training_progress = 0
        st.session_state.loss_history = []
        st.session_state.val_loss_history = []
        st.session_state.metric_history = {}
        st.session_state.epochs = []
    
    # Display current progress
    with progress_placeholder.container():
        if st.session_state.training_progress < 100:
            st.progress(st.session_state.training_progress / 100)
            st.write(f"Training Progress: {st.session_state.training_progress}%")
        else:
            st.success("Training Complete!")
    
    # Display loss chart
    with loss_chart_placeholder.container():
        if st.session_state.loss_history:
            # Create loss chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=st.session_state.epochs,
                y=st.session_state.loss_history,
                name="Training Loss",
                mode="lines+markers"
            ))
            
            if st.session_state.val_loss_history:
                fig.add_trace(go.Scatter(
                    x=st.session_state.epochs,
                    y=st.session_state.val_loss_history,
                    name="Validation Loss",
                    mode="lines+markers",
                    line=dict(dash="dash")
                ))
            
            fig.update_layout(
                title="Training & Validation Loss",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training data available yet.")
    
    # Display metrics
    with metrics_placeholder.container():
        if st.session_state.metric_history:
            # Create columns for metrics
            metric_cols = st.columns(len(st.session_state.metric_history))
            
            # Display each metric
            for i, (metric_name, metric_values) in enumerate(st.session_state.metric_history.items()):
                with metric_cols[i]:
                    # Display current value
                    if metric_values:
                        current_value = metric_values[-1]
                        st.metric(
                            label=metric_name.replace("_", " ").title(),
                            value=f"{current_value:.4f}"
                        )
        else:
            st.info("No metrics available yet.")
    
    # Demo mode: Simulate training progress
    if st.button("Simulate Training"):
        simulate_training_progress()

def simulate_training_progress():
    """
    Simulates training progress for demonstration purposes.
    """
    # Reset progress
    st.session_state.training_progress = 0
    st.session_state.loss_history = []
    st.session_state.val_loss_history = []
    st.session_state.metric_history = {
        "accuracy": [],
        "precision": [],
        "recall": []
    }
    st.session_state.epochs = []
    
    # Simulate epochs
    total_epochs = 10
    
    for epoch in range(total_epochs):
        # Update progress
        st.session_state.training_progress = (epoch + 1) / total_epochs * 100
        st.session_state.epochs.append(epoch + 1)
        
        # Generate loss values (decreasing over time with some noise)
        base_loss = 1.0 - 0.08 * epoch
        train_loss = max(0.1, base_loss + np.random.normal(0, 0.05))
        val_loss = max(0.15, base_loss + 0.1 + np.random.normal(0, 0.07))
        
        st.session_state.loss_history.append(train_loss)
        st.session_state.val_loss_history.append(val_loss)
        
        # Generate metric values (increasing over time with some noise)
        accuracy = min(0.95, 0.5 + 0.04 * epoch + np.random.normal(0, 0.02))
        precision = min(0.93, 0.45 + 0.05 * epoch + np.random.normal(0, 0.03))
        recall = min(0.94, 0.4 + 0.06 * epoch + np.random.normal(0, 0.025))
        
        st.session_state.metric_history["accuracy"].append(accuracy)
        st.session_state.metric_history["precision"].append(precision)
        st.session_state.metric_history["recall"].append(recall)
        
        # Rerun to update visualizations
        time.sleep(0.5)
        st.rerun()

def create_optimization_monitor():
    """
    Creates a real-time visualization for hyperparameter optimization.
    """
    st.markdown("## Optimization Monitor")
    
    # Create placeholders for visualizations
    process_placeholder = st.empty()
    best_results_placeholder = st.empty()
    param_charts_placeholder = st.empty()
    
    # Initialize session state for optimization data if not already done
    if 'optimization_rounds' not in st.session_state:
        st.session_state.optimization_rounds = 0
        st.session_state.trials = []
        st.session_state.best_score = 0
        st.session_state.best_params = {}
    
    # Display optimization process
    with process_placeholder.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Trials Completed: {st.session_state.optimization_rounds}")
            
            if st.session_state.optimization_rounds > 0:
                # Create trials visualization
                trials_df = pd.DataFrame(st.session_state.trials)
                
                fig = px.scatter(
                    trials_df,
                    x="trial_id",
                    y="score",
                    color="score",
                    size="score",
                    color_continuous_scale="Viridis",
                    title="Trial Scores"
                )
                
                # Add best score line
                fig.add_hline(
                    y=st.session_state.best_score,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Best: {st.session_state.best_score:.4f}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No optimization trials completed yet.")
        
        with col2:
            if st.session_state.trials:
                # Create parallel coordinates plot for parameters
                param_cols = [col for col in trials_df.columns if col.startswith("param_")]
                
                if param_cols:
                    fig = px.parallel_coordinates(
                        trials_df,
                        dimensions=param_cols + ["score"],
                        color="score",
                        color_continuous_scale="Viridis",
                        title="Parameter Space Exploration"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Display best results
    with best_results_placeholder.container():
        if st.session_state.best_params:
            st.markdown("### Best Configuration")
            
            # Create columns for parameters
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Best Score:** {st.session_state.best_score:.4f}")
                st.markdown(f"**Found at Trial:** {st.session_state.best_trial}")
            
            with col2:
                # Display best parameters
                st.markdown("**Best Parameters:**")
                for param, value in st.session_state.best_params.items():
                    st.markdown(f"- {param}: {value}")
    
    # Parameter importance visualization
    with param_charts_placeholder.container():
        if st.session_state.trials and len(st.session_state.trials) >= 5:
            st.markdown("### Parameter Importance")
            
            # Calculate parameter importance (simplified)
            param_cols = [col for col in trials_df.columns if col.startswith("param_")]
            importance = {}
            
            for param in param_cols:
                # Calculate correlation with score as a simple importance metric
                correlation = abs(trials_df[param].corr(trials_df["score"]))
                importance[param.replace("param_", "")] = correlation
            
            # Create importance chart
            importance_df = pd.DataFrame({
                "Parameter": list(importance.keys()),
                "Importance": list(importance.values())
            }).sort_values("Importance", ascending=False)
            
            fig = px.bar(
                importance_df,
                x="Parameter",
                y="Importance",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Parameter Importance (Correlation with Score)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Demo mode: Simulate optimization
    if st.button("Simulate Optimization"):
        simulate_optimization()

def simulate_optimization():
    """
    Simulates hyperparameter optimization for demonstration purposes.
    """
    # Reset optimization data
    st.session_state.optimization_rounds = 0
    st.session_state.trials = []
    st.session_state.best_score = 0
    st.session_state.best_params = {}
    st.session_state.best_trial = 0
    
    # Define parameter space
    param_space = {
        "learning_rate": (0.001, 0.1),
        "max_depth": (3, 10),
        "n_estimators": (50, 200)
    }
    
    # Simulate optimization rounds
    total_rounds = 15
    
    for round_id in range(total_rounds):
        # Generate random parameters
        params = {
            "learning_rate": np.random.uniform(param_space["learning_rate"][0], param_space["learning_rate"][1]),
            "max_depth": int(np.random.uniform(param_space["max_depth"][0], param_space["max_depth"][1])),
            "n_estimators": int(np.random.uniform(param_space["n_estimators"][0], param_space["n_estimators"][1]))
        }
        
        # Calculate a score based on parameters
        # This is a simplified model of how parameters affect performance
        score = (
            0.75 + 
            0.15 * (params["learning_rate"] - 0.001) / (0.1 - 0.001) * np.exp(-5 * params["learning_rate"]) +
            0.05 * (params["max_depth"] - 3) / (10 - 3) +
            0.05 * (params["n_estimators"] - 50) / (200 - 50)
        )
        
        # Add some noise to the score
        score += np.random.normal(0, 0.01)
        score = min(0.99, max(0.5, score))
        
        # Update best score and params if necessary
        if score > st.session_state.best_score:
            st.session_state.best_score = score
            st.session_state.best_params = params.copy()
            st.session_state.best_trial = round_id + 1
        
        # Add trial to history
        trial = {
            "trial_id": round_id + 1,
            "score": score,
            "param_learning_rate": params["learning_rate"],
            "param_max_depth": params["max_depth"],
            "param_n_estimators": params["n_estimators"]
        }
        
        st.session_state.trials.append(trial)
        st.session_state.optimization_rounds += 1
        
        # Rerun to update visualizations
        time.sleep(0.3)
        st.rerun()

def create_convergence_monitor():
    """
    Creates a visualization for monitoring model convergence during training.
    """
    st.markdown("## Convergence Monitor")
    
    # Create placeholders for visualizations
    convergence_placeholder = st.empty()
    weight_dist_placeholder = st.empty()
    gradient_placeholder = st.empty()
    
    # Initialize session state for convergence data if not already done
    if 'convergence_data' not in st.session_state:
        st.session_state.convergence_data = {
            "iterations": [],
            "loss": [],
            "gradient_norm": [],
            "weight_change": [],
            "weight_distributions": []
        }
    
    # Display convergence metrics
    with convergence_placeholder.container():
        if st.session_state.convergence_data["iterations"]:
            # Create convergence metrics visualization
            fig = go.Figure()
            
            # Add loss curve
            fig.add_trace(go.Scatter(
                x=st.session_state.convergence_data["iterations"],
                y=st.session_state.convergence_data["loss"],
                name="Loss",
                mode="lines+markers",
                yaxis="y"
            ))
            
            # Add gradient norm
            fig.add_trace(go.Scatter(
                x=st.session_state.convergence_data["iterations"],
                y=st.session_state.convergence_data["gradient_norm"],
                name="Gradient Norm",
                mode="lines+markers",
                yaxis="y2"
            ))
            
            # Add weight change
            fig.add_trace(go.Scatter(
                x=st.session_state.convergence_data["iterations"],
                y=st.session_state.convergence_data["weight_change"],
                name="Weight Change",
                mode="lines+markers",
                yaxis="y3"
            ))
            
            # Set up layout with multiple Y axes
            fig.update_layout(
                title="Training Convergence Metrics",
                xaxis=dict(title="Iteration"),
                yaxis=dict(
                    title="Loss",
                    side="left"
                ),
                yaxis2=dict(
                    title="Gradient Norm",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                yaxis3=dict(
                    title="Weight Change",
                    overlaying="y",
                    side="right",
                    position=0.85,
                    showgrid=False
                ),
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No convergence data available yet.")
    
    # Display weight distribution
    with weight_dist_placeholder.container():
        if st.session_state.convergence_data["weight_distributions"]:
            st.markdown("### Weight Distribution Evolution")
            
            # Create weight distribution visualization
            iterations = st.session_state.convergence_data["iterations"]
            weight_distributions = st.session_state.convergence_data["weight_distributions"]
            
            # Let user select which iterations to compare
            if len(iterations) > 1:
                iteration_options = [f"Iteration {it}" for it in iterations]
                selected_iterations = st.multiselect(
                    "Select iterations to compare:",
                    iteration_options,
                    default=[iteration_options[0], iteration_options[-1]]
                )
                
                if selected_iterations:
                    # Create distribution plot
                    fig = go.Figure()
                    
                    for selected in selected_iterations:
                        # Extract iteration number
                        it_idx = iteration_options.index(selected)
                        it_num = iterations[it_idx]
                        
                        # Add distribution
                        fig.add_trace(go.Histogram(
                            x=weight_distributions[it_idx],
                            name=f"Iteration {it_num}",
                            opacity=0.7,
                            nbinsx=30
                        ))
                    
                    fig.update_layout(
                        title="Weight Distribution Comparison",
                        xaxis_title="Weight Value",
                        yaxis_title="Frequency",
                        barmode="overlay"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need more iterations to compare weight distributions.")
    
    # Display gradient visualization
    with gradient_placeholder.container():
        st.markdown("### Gradient Evolution")
        st.info("Gradient visualization would be displayed here in a real implementation.")
    
    # Demo mode: Simulate convergence
    if st.button("Simulate Convergence"):
        simulate_convergence()

def simulate_convergence():
    """
    Simulates model convergence for demonstration purposes.
    """
    # Reset convergence data
    st.session_state.convergence_data = {
        "iterations": [],
        "loss": [],
        "gradient_norm": [],
        "weight_change": [],
        "weight_distributions": []
    }
    
    # Simulate iterations
    total_iterations = 20
    
    # Initial weight distribution
    np.random.seed(42)
    weights = np.random.normal(0, 1, 1000)
    
    for iteration in range(1, total_iterations + 1):
        # Update iteration counter
        st.session_state.convergence_data["iterations"].append(iteration)
        
        # Generate loss (decreasing exponentially with noise)
        loss = 2.0 * np.exp(-0.2 * iteration) + 0.1 + np.random.normal(0, 0.05)
        st.session_state.convergence_data["loss"].append(loss)
        
        # Generate gradient norm (decreasing with noise)
        grad_norm = 1.0 * np.exp(-0.15 * iteration) + 0.05 + np.random.normal(0, 0.02)
        st.session_state.convergence_data["gradient_norm"].append(grad_norm)
        
        # Update weights (simulate optimization step)
        weight_updates = np.random.normal(0, grad_norm / 5, len(weights))
        weights = weights - weight_updates
        
        # Calculate weight change
        weight_change = np.linalg.norm(weight_updates)
        st.session_state.convergence_data["weight_change"].append(weight_change)
        
        # Store weight distribution snapshot
        st.session_state.convergence_data["weight_distributions"].append(weights.copy())
        
        # Rerun to update visualizations
        time.sleep(0.2)
        st.rerun()
