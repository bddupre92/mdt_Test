"""
Component Details Visualization Module.

This module provides detailed visualizations for each component in the pipeline 
when a user clicks on a component in the interactive pipeline visualization.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Try to import from local modules
try:
    from visualization.expert_viz import visualize_expert_contributions
    from visualization.validation_viz import display_validation_metrics
    from visualization.performance_dashboard import create_performance_dashboard
    from visualization.gating_network_viz import visualize_gating_decisions
    from visualization.moe_visualizer import visualize_moe_integration
except ImportError:
    # Define placeholder functions if modules are not available
    def visualize_expert_contributions(*args, **kwargs):
        return None
    
    def display_validation_metrics(*args, **kwargs):
        return None
    
    def create_performance_dashboard(*args, **kwargs):
        return None
    
    def visualize_gating_decisions(*args, **kwargs):
        return None
    
    def visualize_moe_integration(*args, **kwargs):
        return None

def load_workflow_data(workflow_id: str = None) -> Dict:
    """
    Load workflow data for visualization.
    
    Args:
        workflow_id: ID of the workflow to load, or None for latest
        
    Returns:
        Dict: Workflow data
    """
    # Default empty workflow data
    default_data = {
        "id": workflow_id or "sample_workflow",
        "timestamp": "2023-01-01T00:00:00",
        "status": "completed",
        "components": {},
        "metrics": {}
    }
    
    # If no workflow_id specified, return default
    if not workflow_id or workflow_id == "sample_workflow":
        return default_data
    
    # Check for workflow in the tracking directory
    tracking_dir = Path(".workflow_tracking")
    if not tracking_dir.exists():
        return default_data
    
    # Look for the workflow file
    workflow_file = tracking_dir / f"workflow_{workflow_id}.json"
    if not workflow_file.exists():
        # Try to find any workflow file if specific ID not found
        workflow_files = list(tracking_dir.glob("workflow_*.json"))
        if not workflow_files:
            return default_data
        workflow_file = workflow_files[0]
    
    # Load the workflow data
    try:
        with open(workflow_file, "r") as f:
            workflow_data = json.load(f)
        return workflow_data
    except Exception as e:
        st.error(f"Error loading workflow data: {e}")
        return default_data

def create_sample_data(component_name: str) -> Dict:
    """
    Create sample data for demonstration when real data is not available.
    
    Args:
        component_name: Name of the component to create sample data for
        
    Returns:
        Dict: Sample data for the component
    """
    # Base sample data structure
    sample_data = {
        "timestamp": "2023-01-01T00:00:00",
        "metrics": {},
        "data_samples": {},
        "visualizations": {}
    }
    
    # Component-specific sample data
    if component_name == "data_preprocessing":
        # Sample metrics for data preprocessing
        sample_data["metrics"] = {
            "completeness": 0.95,
            "consistency": 0.88,
            "processing_time_ms": 235,
            "outliers_removed": 12,
            "records_processed": 1000
        }
        
        # Sample data distributions
        np.random.seed(42)
        raw_data = np.random.normal(0, 1, 1000)
        processed_data = np.random.normal(0, 0.8, 1000)
        
        sample_data["data_samples"] = {
            "raw_data": raw_data.tolist()[:5],
            "processed_data": processed_data.tolist()[:5]
        }
        
        sample_data["visualizations"] = {
            "distribution_before": {
                "type": "histogram",
                "data": raw_data.tolist()
            },
            "distribution_after": {
                "type": "histogram",
                "data": processed_data.tolist()
            }
        }
    
    elif component_name == "feature_extraction":
        # Sample metrics for feature extraction
        sample_data["metrics"] = {
            "features_extracted": 10,
            "feature_extraction_time_ms": 150,
            "dimensionality_reduction": "PCA",
            "variance_explained": 0.85
        }
        
        # Sample feature importance
        feature_names = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
        feature_importance = [0.35, 0.25, 0.18, 0.12, 0.1]
        
        sample_data["data_samples"] = {
            "feature_names": feature_names,
            "feature_importance": feature_importance
        }
        
        sample_data["visualizations"] = {
            "feature_importance": {
                "type": "bar",
                "x": feature_names,
                "y": feature_importance
            },
            "pca_visualization": {
                "type": "scatter",
                "data": np.random.rand(100, 2).tolist()
            }
        }
    
    elif component_name == "missing_data_handling":
        # Sample metrics for missing data handling
        sample_data["metrics"] = {
            "missing_values_before": 120,
            "missing_values_after": 0,
            "imputation_accuracy": 0.89,
            "imputation_time_ms": 175
        }
        
        # Sample missing data patterns
        missing_patterns = np.zeros((10, 5))
        for i in range(10):
            missing_patterns[i, np.random.choice(5, 2, replace=False)] = 1
        
        sample_data["data_samples"] = {
            "missing_patterns": missing_patterns.tolist()
        }
        
        sample_data["visualizations"] = {
            "missing_heatmap": {
                "type": "heatmap",
                "data": missing_patterns.tolist()
            },
            "imputation_accuracy": {
                "type": "bar",
                "x": ["MICE", "KNN", "Mean", "Median", "Mode"],
                "y": [0.89, 0.85, 0.78, 0.77, 0.72]
            }
        }
    
    elif component_name == "expert_training":
        # Sample metrics for expert training
        sample_data["metrics"] = {
            "num_experts": 3,
            "training_time_s": 45.2,
            "validation_accuracy": 0.87,
            "experts_converged": True
        }
        
        # Sample training history
        epochs = list(range(1, 11))
        train_loss = [0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25, 0.24]
        val_loss = [0.85, 0.7, 0.6, 0.55, 0.5, 0.48, 0.47, 0.47, 0.46, 0.46]
        
        sample_data["data_samples"] = {
            "expert_architectures": ["MLP", "CNN", "LSTM"],
            "expert_parameters": [12500, 28900, 15300]
        }
        
        sample_data["visualizations"] = {
            "training_history": {
                "type": "line",
                "x": epochs,
                "y": [train_loss, val_loss],
                "names": ["Training Loss", "Validation Loss"]
            },
            "expert_comparison": {
                "type": "bar",
                "x": ["Expert 1", "Expert 2", "Expert 3"],
                "y": [0.86, 0.82, 0.85]
            }
        }
    
    elif component_name == "gating_network":
        # Sample metrics for gating network
        sample_data["metrics"] = {
            "gating_accuracy": 0.91,
            "avg_confidence": 0.88,
            "routing_entropy": 0.72,
            "training_time_s": 28.5
        }
        
        # Sample gating decisions
        np.random.seed(42)
        gating_decisions = np.random.rand(100, 3)
        gating_decisions = gating_decisions / gating_decisions.sum(axis=1, keepdims=True)
        
        sample_data["data_samples"] = {
            "gating_weights_sample": gating_decisions[:5].tolist()
        }
        
        sample_data["visualizations"] = {
            "gating_heatmap": {
                "type": "heatmap",
                "data": gating_decisions[:20].tolist()
            },
            "expert_usage": {
                "type": "pie",
                "values": [0.4, 0.35, 0.25],
                "labels": ["Expert 1", "Expert 2", "Expert 3"]
            }
        }
    
    elif component_name == "moe_integration":
        # Sample metrics for MoE integration
        sample_data["metrics"] = {
            "ensemble_improvement": 0.08,
            "integration_time_ms": 14.5,
            "final_accuracy": 0.92,
            "confidence_calibration": 0.95
        }
        
        # Sample integration weights
        sample_data["data_samples"] = {
            "integration_weights": [0.45, 0.35, 0.2],
            "expert_contributions": [42, 35, 23]
        }
        
        sample_data["visualizations"] = {
            "performance_comparison": {
                "type": "bar",
                "x": ["Expert 1", "Expert 2", "Expert 3", "Ensemble"],
                "y": [0.86, 0.82, 0.85, 0.92]
            },
            "contribution_pie": {
                "type": "pie",
                "values": [42, 35, 23],
                "labels": ["Expert 1", "Expert 2", "Expert 3"]
            }
        }
    
    elif component_name == "output_generation":
        # Sample metrics for output generation
        sample_data["metrics"] = {
            "final_accuracy": 0.93,
            "f1_score": 0.91,
            "output_generation_time_ms": 8.2,
            "output_size_kb": 24.5
        }
        
        # Sample predictions
        sample_data["data_samples"] = {
            "predictions_sample": np.random.randn(5).tolist(),
            "ground_truth_sample": np.random.randn(5).tolist()
        }
        
        sample_data["visualizations"] = {
            "prediction_vs_actual": {
                "type": "scatter",
                "x": np.random.randn(50).tolist(),
                "y": np.random.randn(50).tolist()
            },
            "performance_metrics": {
                "type": "radar",
                "metrics": ["Accuracy", "Precision", "Recall", "F1", "AUC"],
                "values": [0.93, 0.92, 0.9, 0.91, 0.94]
            }
        }
    
    return sample_data

def render_data_preprocessing_details(workflow_data: Dict):
    """Render detailed visualizations for data preprocessing component."""
    st.markdown("## Data Preprocessing Details")
    
    # Get component data or create sample
    comp_data = workflow_data.get("components", {}).get("data_preprocessing", {})
    if not comp_data:
        st.warning("No data preprocessing details found in workflow. Using sample data.")
        comp_data = create_sample_data("data_preprocessing")
    
    # Display metrics
    metrics = comp_data.get("metrics", {})
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Completeness", f"{metrics.get('completeness', 0)*100:.1f}%")
        with col2:
            st.metric("Consistency", f"{metrics.get('consistency', 0)*100:.1f}%")
        with col3:
            st.metric("Processing Time", f"{metrics.get('processing_time_ms', 0)/1000:.2f}s")
        with col4:
            st.metric("Records Processed", metrics.get("records_processed", 0))
    
    # Data distribution visualization
    st.markdown("### Data Distribution Before/After Preprocessing")
    
    # Get visualization data or use sample
    viz_data = comp_data.get("visualizations", {})
    
    if "distribution_before" in viz_data and "distribution_after" in viz_data:
        before_data = viz_data["distribution_before"].get("data", [])
        after_data = viz_data["distribution_after"].get("data", [])
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=before_data, name="Before", opacity=0.7))
        fig.add_trace(go.Histogram(x=after_data, name="After", opacity=0.7))
        fig.update_layout(barmode="overlay", title="Data Distribution Comparison")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No distribution visualization data available.")
        
        # Create sample visualization
        np.random.seed(42)
        before = np.random.normal(0, 1, 1000)
        after = np.random.normal(0, 0.8, 1000)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=before, name="Before", opacity=0.7))
        fig.add_trace(go.Histogram(x=after, name="After", opacity=0.7))
        fig.update_layout(barmode="overlay", title="Sample Data Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Data samples
    st.markdown("### Data Samples")
    data_samples = comp_data.get("data_samples", {})
    
    if data_samples:
        raw_data = data_samples.get("raw_data", [])
        processed_data = data_samples.get("processed_data", [])
        
        if raw_data and processed_data:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Raw Data")
                st.write(pd.DataFrame({"Value": raw_data}))
            with col2:
                st.markdown("#### Processed Data")
                st.write(pd.DataFrame({"Value": processed_data}))
        else:
            st.info("No data samples available.")
    else:
        st.info("No data samples available.")
    
    # Interactive data input section
    st.markdown("### Try with Your Own Data")
    
    with st.expander("Upload your own data for preprocessing"):
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.write(data.head())
                
                if st.button("Preprocess Data"):
                    # Simulate preprocessing
                    st.success("Data preprocessing completed!")
                    
                    # Show simulated results
                    processed_data = data.copy()
                    # Normalize numeric columns
                    for col in processed_data.select_dtypes(include=['float64', 'int64']).columns:
                        if processed_data[col].std() > 0:
                            processed_data[col] = (processed_data[col] - processed_data[col].mean()) / processed_data[col].std()
                    
                    st.write("Processed Data Preview:")
                    st.write(processed_data.head())
                    
                    # Download button for processed data
                    st.download_button(
                        label="Download Processed Data",
                        data=processed_data.to_csv(index=False).encode("utf-8"),
                        file_name="processed_data.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing file: {e}")

def render_feature_extraction_details(workflow_data: Dict):
    """Render detailed visualizations for feature extraction component."""
    st.markdown("## Feature Extraction Details")
    
    # Get component data or create sample
    comp_data = workflow_data.get("components", {}).get("feature_extraction", {})
    if not comp_data:
        st.warning("No feature extraction details found in workflow. Using sample data.")
        comp_data = create_sample_data("feature_extraction")
    
    # Display metrics
    metrics = comp_data.get("metrics", {})
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Features Extracted", metrics.get("features_extracted", 0))
        with col2:
            st.metric("Extraction Time", f"{metrics.get('feature_extraction_time_ms', 0)/1000:.2f}s")
        with col3:
            st.metric("Dim. Reduction", metrics.get("dimensionality_reduction", "None"))
        with col4:
            st.metric("Variance Explained", f"{metrics.get('variance_explained', 0)*100:.1f}%")
    
    # Feature importance visualization
    st.markdown("### Feature Importance")
    
    viz_data = comp_data.get("visualizations", {})
    data_samples = comp_data.get("data_samples", {})
    
    if "feature_importance" in viz_data:
        fi_viz = viz_data["feature_importance"]
        feature_names = fi_viz.get("x", data_samples.get("feature_names", []))
        feature_importance = fi_viz.get("y", data_samples.get("feature_importance", []))
        
        if feature_names and feature_importance:
            fig = go.Figure(go.Bar(
                x=feature_importance,
                y=feature_names,
                orientation="h"
            ))
            fig.update_layout(title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature importance data available.")
    else:
        # Use sample data from data_samples if available
        feature_names = data_samples.get("feature_names", [])
        feature_importance = data_samples.get("feature_importance", [])
        
        if feature_names and feature_importance:
            fig = go.Figure(go.Bar(
                x=feature_importance,
                y=feature_names,
                orientation="h"
            ))
            fig.update_layout(title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature importance data available.")
    
    # PCA or t-SNE visualization if available
    st.markdown("### Dimensionality Reduction")
    
    if "pca_visualization" in viz_data:
        pca_data = viz_data["pca_visualization"].get("data", [])
        
        if pca_data:
            # Convert to numpy array for easier handling
            pca_data = np.array(pca_data)
            
            # Create scatter plot
            fig = px.scatter(
                x=pca_data[:, 0],
                y=pca_data[:, 1],
                title="PCA Visualization"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No PCA visualization data available.")
    else:
        st.info("No dimensionality reduction visualization available.")
    
    # Interactive feature selection
    st.markdown("### Interactive Feature Selection")
    
    with st.expander("Select features for your model"):
        # Sample features to select from
        available_features = feature_names if feature_names else [
            "Feature A", "Feature B", "Feature C", "Feature D", 
            "Feature E", "Feature F", "Feature G", "Feature H"
        ]
        
        selected_features = st.multiselect(
            "Select features to include",
            available_features,
            default=available_features[:3]
        )
        
        if st.button("Generate Feature Subset"):
            if selected_features:
                st.success(f"Feature subset created with {len(selected_features)} features!")
                
                # Display selected features and their importance if available
                if feature_names and feature_importance and set(selected_features).issubset(set(feature_names)):
                    # Create mapping of feature names to importance
                    importance_map = {name: imp for name, imp in zip(feature_names, feature_importance)}
                    
                    # Get importance for selected features
                    selected_importance = [importance_map.get(feat, 0) for feat in selected_features]
                    
                    # Create visualization
                    fig = go.Figure(go.Bar(
                        x=selected_importance,
                        y=selected_features,
                        orientation="h"
                    ))
                    fig.update_layout(title="Selected Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Create dummy visualization for selected features
                    selected_importance = np.random.uniform(0.1, 0.5, len(selected_features))
                    
                    fig = go.Figure(go.Bar(
                        x=selected_importance,
                        y=selected_features,
                        orientation="h"
                    ))
                    fig.update_layout(title="Selected Feature Importance (Simulated)")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one feature.")

# Add more component render functions here for other pipeline components

def render_component_details(component_name, component_data):
    """
    Render detailed visualizations for a specific pipeline component.
    
    Args:
        component_name (str): Name of the component to visualize
        component_data (dict): Data for the component to visualize
    """
    st.markdown(f"## {component_name.replace('_', ' ').title()} Details")
    
    # Add tabs for different types of information
    tabs = st.tabs(["Metrics", "Visualizations", "Data Flow"])
    
    # First tab: Metrics
    with tabs[0]:
        render_component_metrics(component_name, component_data)
    
    # Second tab: Visualizations
    with tabs[1]:
        # Render component-specific visualizations based on component type
        if component_name == "data_preprocessing":
            render_data_preprocessing_visualization(component_data)
        elif component_name == "feature_extraction":
            render_feature_extraction_visualization(component_data)
        elif component_name == "missing_data_handling":
            render_missing_data_visualization(component_data)
        elif component_name == "expert_training":
            render_expert_training_visualization(component_data)
        elif component_name == "gating_network":
            render_gating_network_visualization(component_data)
        elif component_name == "moe_integration":
            render_moe_integration_visualization(component_data)
        elif component_name == "output_generation":
            render_output_visualization(component_data)
        else:
            st.info(f"No specific visualizations available for {component_name}.")
    
    # Third tab: Data Flow
    with tabs[2]:
        # Show input and output data
        if component_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Input Data")
                input_data = component_data.get("input", None)
                if isinstance(input_data, pd.DataFrame):
                    st.dataframe(input_data.head())
                elif isinstance(input_data, dict):
                    st.json(input_data)
                elif input_data is not None:
                    st.write(input_data)
                else:
                    st.info("No input data available.")
            
            with col2:
                st.markdown("### Output Data")
                output_data = component_data.get("output", None)
                if isinstance(output_data, pd.DataFrame):
                    st.dataframe(output_data.head())
                elif isinstance(output_data, dict):
                    st.json(output_data)
                elif output_data is not None:
                    st.write(output_data)
                else:
                    st.info("No output data available.")
        else:
            st.info("No data flow information available for this component.")

def render_component_metrics(component_name: str, component_data: dict = None):
    """
    Render metrics for a specific pipeline component.
    
    Args:
        component_name: Name of the component
        component_data: Component data if available
    """
    st.subheader(f"{component_name} Metrics")
    
    if component_data is None:
        st.info("No metrics data available for this component. Run the pipeline first.")
        return
        
    # Check if we're in demo mode and need to generate sample metrics
    if st.session_state.get('demo_mode') and not component_data.get('metrics'):
        if st.button("Generate Sample Metrics"):
            st.session_state['generating_metrics'] = True
            with st.spinner(f"Generating sample metrics for {component_name}..."):
                sample_metrics = generate_sample_metrics(component_name)
                if component_data:
                    component_data['metrics'] = sample_metrics
                else:
                    component_data = {'metrics': sample_metrics}
                st.session_state['component_data'] = component_data
                # Use the new st.rerun() instead of the deprecated experimental_rerun
                st.rerun()
        return

    # We have metrics data, display it
    metrics = component_data["metrics"]
    
    if isinstance(metrics, dict) and metrics:
        # Create columns for metric cards
        cols = st.columns(3)
        
        # Display key metrics as cards
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with cols[i % 3]:
                st.metric(
                    label=metric_name.replace("_", " ").title(),
                    value=metric_value if isinstance(metric_value, (str, int)) else f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                )
        
        # Create a detailed metrics table
        st.markdown("#### Detailed Metrics")
        
        # Convert all values to strings for displaying in the dataframe
        # This prevents PyArrow conversion issues with mixed types
        metrics_str = {k: str(v) for k, v in metrics.items()}
        
        metrics_df = pd.DataFrame({
            "Metric": list(metrics_str.keys()),
            "Value": list(metrics_str.values())
        })
        st.dataframe(metrics_df, hide_index=True)
        
        # If we have numeric metrics, create visualizations
        numeric_metrics = {k: v for k, v in metrics.items() 
                         if isinstance(v, (int, float)) and not isinstance(v, bool)}
        
        if numeric_metrics:
            # Create bar chart for visualization
            st.markdown("#### Metrics Visualization")
            
            fig = px.bar(
                x=list(numeric_metrics.keys()),
                y=list(numeric_metrics.values()),
                labels={"x": "Metric", "y": "Value"},
                title=f"Performance Metrics for {component_name.replace('_', ' ').title()}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No detailed metrics found for {component_name}.")

def render_data_preprocessing_visualization(component_data):
    """Render visualizations for data preprocessing component."""
    if not component_data or "input" not in component_data or "output" not in component_data:
        st.info("No preprocessing data available to visualize.")
        return
    
    # Get input and output data
    input_data = component_data.get("input", {})
    output_data = component_data.get("output", {})
    
    # Basic data stats comparison
    if isinstance(input_data, pd.DataFrame) and isinstance(output_data, pd.DataFrame):
        st.markdown("#### Data Statistics Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Input Data Stats**")
            st.dataframe(input_data.describe())
        
        with col2:
            st.markdown("**Output Data Stats**")
            st.dataframe(output_data.describe())
        
        # Distribution visualization for numeric columns
        st.markdown("#### Data Distributions")
        
        # Select a column to visualize
        numeric_columns = input_data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            selected_column = st.selectbox("Select column to visualize:", numeric_columns)
            
            # Create histogram comparing before/after
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=input_data[selected_column], 
                name="Before Processing",
                opacity=0.7
            ))
            fig.add_trace(go.Histogram(
                x=output_data[selected_column] if selected_column in output_data.columns else None, 
                name="After Processing",
                opacity=0.7
            ))
            fig.update_layout(
                barmode="overlay",
                title=f"Distribution of {selected_column} Before/After Processing",
                xaxis_title=selected_column,
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Missing values visualization
    st.markdown("#### Missing Values")
    
    if isinstance(input_data, pd.DataFrame):
        # Calculate missing values before/after
        missing_before = input_data.isnull().sum()
        missing_after = output_data.isnull().sum() if isinstance(output_data, pd.DataFrame) else pd.Series()
        
        # Create comparison dataframe
        missing_df = pd.DataFrame({
            "Missing Before": missing_before,
            "Missing After": missing_after.reindex(missing_before.index, fill_value=0)
        }).sort_values("Missing Before", ascending=False)
        
        # Filter to columns that had missing values
        missing_df = missing_df[missing_df["Missing Before"] > 0]
        
        if not missing_df.empty:
            # Create bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=missing_df.index,
                x=missing_df["Missing Before"],
                name="Before Processing",
                orientation="h"
            ))
            fig.add_trace(go.Bar(
                y=missing_df.index,
                x=missing_df["Missing After"],
                name="After Processing",
                orientation="h"
            ))
            fig.update_layout(
                barmode="group",
                title="Missing Values Before/After Processing",
                xaxis_title="Count",
                yaxis_title="Column"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values found in the input data.")

def render_feature_extraction_visualization(component_data):
    """Render visualizations for feature extraction component."""
    if not component_data:
        st.info("No feature extraction data available to visualize.")
        return
    
    # Get output data
    output_data = component_data.get("output", {})
    
    # Feature importance visualization
    st.markdown("#### Feature Importance")
    
    # Check if feature importances are available
    feature_importances = component_data.get("feature_importances", {})
    
    if feature_importances:
        # Convert to DataFrame
        if isinstance(feature_importances, dict):
            importance_df = pd.DataFrame({
                "Feature": list(feature_importances.keys()),
                "Importance": list(feature_importances.values())
            })
        else:
            importance_df = pd.DataFrame(feature_importances)
        
        # Sort by importance
        importance_df = importance_df.sort_values("Importance", ascending=False)
        
        # Create bar chart
        fig = go.Figure(go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Feature"],
            orientation="h"
        ))
        fig.update_layout(
            title="Feature Importance",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)
    elif isinstance(output_data, pd.DataFrame):
        st.info("No explicit feature importance data available. Showing extracted features.")
        
        # Show extracted features
        st.dataframe(output_data.head())
    else:
        st.info("No feature importance or extracted feature data available.")
    
    # Feature correlation heatmap
    st.markdown("#### Feature Correlations")
    
    if isinstance(output_data, pd.DataFrame):
        # Calculate correlations
        numeric_data = output_data.select_dtypes(include=[np.number])
        
        if not numeric_data.empty:
            corr = numeric_data.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr,
                labels=dict(color="Correlation"),
                x=corr.columns,
                y=corr.columns,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                width=600, height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric features available for correlation analysis.")
    else:
        st.info("No feature data available for correlation analysis.")

def render_missing_data_visualization(component_data):
    """Render visualizations for missing data handling component."""
    if not component_data:
        st.info("No missing data handling results available to visualize.")
        return
    
    # Get input and output data
    input_data = component_data.get("input", {})
    output_data = component_data.get("output", {})
    
    # Missing data patterns visualization
    st.markdown("#### Missing Data Patterns")
    
    if isinstance(input_data, pd.DataFrame):
        # Create missing data mask
        missing_mask = input_data.isnull()
        
        # Visualize only columns with missing values
        columns_with_missing = missing_mask.columns[missing_mask.any()].tolist()
        
        if columns_with_missing:
            # Take a sample if there are too many rows
            if len(input_data) > 20:
                missing_sample = missing_mask[columns_with_missing].head(20)
            else:
                missing_sample = missing_mask[columns_with_missing]
            
            # Create heatmap
            fig = px.imshow(
                missing_sample.T,
                labels=dict(x="Sample", y="Feature", color="Missing"),
                color_continuous_scale=["white", "red"],
                title="Missing Data Patterns (Red = Missing)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values found in the input data.")
    
    # Imputation accuracy visualization
    st.markdown("#### Imputation Metrics")
    
    # Check for imputation metrics
    imputation_metrics = component_data.get("metrics", {})
    
    if imputation_metrics:
        # Create metrics display
        metrics_to_show = {}
        for k, v in imputation_metrics.items():
            if any(term in k.lower() for term in ["accuracy", "error", "score", "imputation"]):
                metrics_to_show[k] = v
        
        if metrics_to_show:
            metrics_df = pd.DataFrame({
                "Metric": list(metrics_to_show.keys()),
                "Value": list(metrics_to_show.values())
            })
            st.dataframe(metrics_df, hide_index=True)
        else:
            st.info("No specific imputation accuracy metrics available.")
    
    # Imputation method comparison
    st.markdown("#### Imputation Method Comparison")
    
    # Check for imputation method data
    imputation_methods = component_data.get("imputation_methods", {})
    
    if imputation_methods:
        # Create bar chart
        method_df = pd.DataFrame({
            "Method": list(imputation_methods.keys()),
            "Accuracy": list(imputation_methods.values())
        })
        
        fig = go.Figure(go.Bar(
            x=method_df["Method"],
            y=method_df["Accuracy"],
            text=[f"{val:.2f}" for val in method_df["Accuracy"]],
            textposition="auto"
        ))
        fig.update_layout(
            title="Imputation Method Accuracy Comparison",
            yaxis=dict(title="Accuracy")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No imputation method comparison data available.")

def render_expert_training_visualization(component_data):
    """Render visualizations for expert training component."""
    if not component_data:
        st.info("No expert training data available to visualize.")
        return
    
    # Extract training history data
    training_history = component_data.get("training_history", {})
    
    if training_history:
        st.markdown("#### Training History")
        
        # Expert selection dropdown
        expert_keys = list(training_history.keys())
        selected_expert = st.selectbox("Select expert:", expert_keys)
        
        if selected_expert in training_history:
            expert_data = training_history[selected_expert]
            
            # Create line chart with training and validation metrics
            fig = go.Figure()
            
            if "train_loss" in expert_data and "epochs" in expert_data:
                fig.add_trace(go.Scatter(
                    x=expert_data["epochs"],
                    y=expert_data["train_loss"],
                    mode="lines",
                    name="Training Loss"
                ))
            
            if "val_loss" in expert_data and "epochs" in expert_data:
                fig.add_trace(go.Scatter(
                    x=expert_data["epochs"],
                    y=expert_data["val_loss"],
                    mode="lines",
                    line=dict(dash="dash"),
                    name="Validation Loss"
                ))
            
            # Add accuracy curves if available
            if "train_accuracy" in expert_data and "epochs" in expert_data:
                fig.add_trace(go.Scatter(
                    x=expert_data["epochs"],
                    y=expert_data["train_accuracy"],
                    mode="lines",
                    name="Training Accuracy",
                    yaxis="y2"
                ))
            
            if "val_accuracy" in expert_data and "epochs" in expert_data:
                fig.add_trace(go.Scatter(
                    x=expert_data["epochs"],
                    y=expert_data["val_accuracy"],
                    mode="lines",
                    line=dict(dash="dash"),
                    name="Validation Accuracy",
                    yaxis="y2"
                ))
            
            fig.update_layout(
                title=f"Training History for {selected_expert}",
                xaxis_title="Epoch",
                yaxis=dict(
                    title="Loss",
                    side="left"
                ),
                yaxis2=dict(
                    title="Accuracy",
                    side="right",
                    overlaying="y",
                    rangemode="tozero",
                    range=[0, 1]
                ),
                legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Expert metrics comparison
    st.markdown("#### Expert Performance Comparison")
    
    expert_metrics = component_data.get("expert_metrics", {})
    
    if expert_metrics:
        # Create comparison dataframe
        metrics_to_compare = ["accuracy", "precision", "recall", "f1_score"]
        comparison_data = {}
        
        for expert, metrics in expert_metrics.items():
            expert_row = {}
            for metric in metrics_to_compare:
                if metric in metrics:
                    expert_row[metric] = metrics[metric]
            comparison_data[expert] = expert_row
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data).T
            
            # Create radar chart
            categories = comparison_df.columns.tolist()
            fig = go.Figure()
            
            for expert in comparison_df.index:
                values = comparison_df.loc[expert].tolist()
                # Add closing point
                values_closed = values + [values[0]]
                categories_closed = categories + [categories[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill="toself",
                    name=expert
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Expert Performance Comparison",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Also show as a table
            st.dataframe(comparison_df, use_container_width=True)

def render_gating_network_visualization(component_data):
    """Render visualizations for gating network component."""
    if not component_data:
        st.info("No gating network data available to visualize.")
        return
    
    # Expert distribution visualization
    st.markdown("#### Expert Selection Distribution")
    
    expert_distribution = component_data.get("expert_distribution", {})
    
    if expert_distribution:
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(expert_distribution.keys()),
            values=list(expert_distribution.values()),
            hole=0.3
        )])
        fig.update_layout(title="Expert Usage Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Routing decisions heatmap
    st.markdown("#### Routing Decisions")
    
    routing_decisions = component_data.get("routing_decisions", [])
    
    if routing_decisions:
        # Extract probabilities
        sample_size = min(20, len(routing_decisions))
        samples = routing_decisions[:sample_size]
        
        # Create data for heatmap
        heatmap_data = []
        expert_keys = []
        
        for sample in samples:
            if "expert_probabilities" in sample:
                if not expert_keys:
                    expert_keys = list(sample["expert_probabilities"].keys())
                
                heatmap_data.append([sample["expert_probabilities"].get(key, 0) for key in expert_keys])
        
        if heatmap_data and expert_keys:
            # Create heatmap
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Expert", y="Sample", color="Probability"),
                x=expert_keys,
                y=[f"Sample {i+1}" for i in range(len(heatmap_data))],
                color_continuous_scale="Viridis",
                zmin=0, zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Confidence over time
    st.markdown("#### Routing Confidence Over Time")
    
    confidence_over_time = component_data.get("confidence_over_time", [])
    
    if confidence_over_time:
        # Extract data
        timestamps = [entry.get("timestamp", i) for i, entry in enumerate(confidence_over_time)]
        confidence = [entry.get("confidence", 0) for entry in confidence_over_time]
        entropy = [entry.get("routing_entropy", 0) for entry in confidence_over_time]
        
        # Create line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=confidence,
            mode="lines",
            name="Confidence"
        ))
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=entropy,
            mode="lines",
            name="Entropy",
            yaxis="y2"
        ))
        fig.update_layout(
            title="Routing Confidence and Entropy Over Time",
            xaxis_title="Time",
            yaxis=dict(
                title="Confidence",
                side="left",
                range=[0, 1]
            ),
            yaxis2=dict(
                title="Entropy",
                side="right",
                overlaying="y",
                range=[0, 1]
            ),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)

def render_moe_integration_visualization(component_data):
    """Render visualizations for MoE integration component."""
    if not component_data:
        st.info("No MoE integration data available to visualize.")
        return
    
    # Model comparison visualization
    st.markdown("#### Model Performance Comparison")
    
    model_comparison = component_data.get("model_comparison", {})
    
    if model_comparison:
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(model_comparison.keys()),
                y=list(model_comparison.values()),
                text=[f"{val:.2f}" for val in model_comparison.values()],
                textposition="auto"
            )
        ])
        fig.update_layout(
            title="Model Accuracy Comparison",
            xaxis_title="Model",
            yaxis=dict(
                title="Accuracy",
                range=[0, 1]
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Expert contribution visualization
    st.markdown("#### Expert Contribution Analysis")
    
    expert_contributions = component_data.get("expert_contributions", {})
    
    if expert_contributions:
        # Create stacked bar chart
        input_types = []
        traces = []
        
        # Get all input types
        for expert, contributions in expert_contributions.items():
            for input_type in contributions:
                if input_type not in input_types:
                    input_types.append(input_type)
        
        # Create traces for each expert
        for expert, contributions in expert_contributions.items():
            values = [contributions.get(input_type, 0) for input_type in input_types]
            traces.append(go.Bar(
                name=expert,
                x=input_types,
                y=values
            ))
        
        # Create figure
        fig = go.Figure(data=traces)
        fig.update_layout(
            barmode="stack",
            title="Expert Weights by Input Type",
            xaxis_title="Input Type",
            yaxis_title="Weight"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Integration methods comparison
    st.markdown("#### Integration Method Comparison")
    
    integration_methods = component_data.get("integration_methods", {})
    
    if integration_methods:
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(integration_methods.keys()),
                y=list(integration_methods.values()),
                text=[f"{val:.2f}" for val in integration_methods.values()],
                textposition="auto"
            )
        ])
        fig.update_layout(
            title="Integration Method Comparison",
            xaxis_title="Method",
            yaxis=dict(
                title="Performance",
                range=[0, 1]
            )
        )
        st.plotly_chart(fig, use_container_width=True)

def render_output_visualization(component_data):
    """Render visualizations for output generation component."""
    if not component_data:
        st.info("No output generation data available to visualize.")
        return
    
    # Performance metrics visualization
    st.markdown("#### Performance Metrics")
    
    metrics = component_data.get("metrics", {})
    
    if metrics:
        # Create metrics display
        metrics_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        })
        st.dataframe(metrics_df, hide_index=True)
        
        # Create radar chart for classification metrics
        classification_metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]
        radar_data = {}
        
        for metric in classification_metrics:
            for k, v in metrics.items():
                if metric in k.lower():
                    radar_data[metric] = v
                    break
        
        if radar_data:
            categories = list(radar_data.keys())
            values = list(radar_data.values())
            
            # Add closing point
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                name="Model Performance"
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Model Performance Metrics",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Predictions visualization
    st.markdown("#### Predictions vs Actual Values")
    
    predictions = component_data.get("predictions", [])
    
    if predictions:
        # Extract data
        actual = [p.get("true_value", 0) for p in predictions]
        predicted = [p.get("predicted_value", 0) for p in predictions]
        
        # Create scatter plot
        fig = px.scatter(
            x=actual,
            y=predicted,
            labels={"x": "Actual", "y": "Predicted"},
            title="Predictions vs Actual Values"
        )
        
        # Add perfect prediction line
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash"),
            name="Perfect Prediction"
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix visualization
    st.markdown("#### Confusion Matrix")
    
    confusion_matrix = component_data.get("confusion_matrix", {})
    
    if confusion_matrix:
        if all(k in confusion_matrix for k in ["true_positive", "false_positive", "true_negative", "false_negative"]):
            # Create confusion matrix visualization
            tp = confusion_matrix.get("true_positive", 0)
            fp = confusion_matrix.get("false_positive", 0)
            tn = confusion_matrix.get("true_negative", 0)
            fn = confusion_matrix.get("false_negative", 0)
            
            matrix = [[tp, fp], [fn, tn]]
            
            fig = px.imshow(
                matrix,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Positive", "Negative"],
                y=["Positive", "Negative"],
                text_auto=True,
                color_continuous_scale="Blues"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Accuracy", f"{accuracy:.2f}")
            
            with metrics_col2:
                st.metric("Precision", f"{precision:.2f}")
            
            with metrics_col3:
                st.metric("Recall", f"{recall:.2f}")
            
            with metrics_col4:
                st.metric("F1 Score", f"{f1:.2f}")

if __name__ == "__main__":
    # For testing this module independently
    st.set_page_config(layout="wide", page_title="Component Details")
    
    # Test with a sample component
    component_name = st.selectbox(
        "Select a component to view",
        [
            "data_preprocessing",
            "feature_extraction", 
            "missing_data_handling",
            "expert_training",
            "gating_network",
            "moe_integration",
            "output_generation"
        ]
    )
    
    render_component_details(component_name) 