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
import logging

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

# Import data utilities
from visualization.data_utils import load_component_data

# Set up logging
logger = logging.getLogger(__name__)

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

def render_component_details(component_name: str):
    """
    Render detailed visualizations for a specific pipeline component.
    
    Args:
        component_name: Name of the component to display
    """
    # Get data for this component
    pipeline_id = st.session_state.get('pipeline_id', None)
    component_data = load_component_data(component_name, pipeline_id)
    
    # Create tabs for different views
    overview_tab, metrics_tab, data_tab = st.tabs(["Overview", "Metrics", "Data Flow"])
    
    with overview_tab:
        render_component_overview(component_name, component_data)
    
    with metrics_tab:
        render_component_metrics(component_name, component_data)
    
    with data_tab:
        render_component_data_flow(component_name, component_data)

def render_component_overview(component_name: str, component_data: Dict[str, Any]):
    """
    Render overview information for a component.
    
    Args:
        component_name: Name of the component
        component_data: Data for this component
    """
    # Component descriptions
    descriptions = {
        "data_preprocessing": """
            The **Data Preprocessing** component cleans and transforms raw input data into a format suitable for
            the pipeline. This includes:
            
            - Converting data types
            - Handling missing values
            - Removing outliers
            - Normalizing numerical features
            - Encoding categorical variables
        """,
        
        "feature_extraction": """
            The **Feature Extraction** component identifies and extracts relevant features from the
            preprocessed data. This includes:
            
            - Selecting important features
            - Dimensionality reduction
            - Creating composite features
            - Scaling and transforming features
        """,
        
        "missing_data_handling": """
            The **Missing Data Handling** component applies specialized techniques to address
            missing values in the dataset. This includes:
            
            - Identifying missing data patterns
            - Applying imputation strategies
            - Validating imputed values
            - Providing quality metrics for missing data treatment
        """,
        
        "expert_training": """
            The **Expert Training** component trains the specialized expert models that form
            the basis of the Mixture of Experts architecture. This includes:
            
            - Optimizing hyperparameters for each expert
            - Training each expert on its specialized domain
            - Evaluating expert performance
            - Determining feature importance for each expert
        """,
        
        "gating_network": """
            The **Gating Network** determines how inputs should be routed to different experts.
            This includes:
            
            - Learning the appropriate weights for each expert
            - Determining confidence in expert predictions
            - Optimizing the routing mechanism
            - Providing a basis for the ensemble
        """,
        
        "moe_integration": """
            The **MoE Integration** component combines the outputs from multiple experts into
            a coherent prediction. This includes:
            
            - Weighting expert predictions
            - Combining predictions via ensemble methods
            - Adjusting confidence based on expert agreement
            - Providing a unified output
        """,
        
        "output_generation": """
            The **Output Generation** component produces the final output of the pipeline,
            formatting predictions and providing explanations. This includes:
            
            - Formatting predictions for consumption
            - Generating confidence intervals
            - Providing feature importance information
            - Creating visualizations and explanations
        """
    }
    
    # Display component description
    st.markdown(f"## {component_name.replace('_', ' ').title()}")
    st.markdown(descriptions.get(component_name, "No description available for this component."))
    
    # Display execution statistics if available
    if "execution_time" in component_data:
        st.markdown(f"**Execution Time:** {component_data['execution_time']:.2f} seconds")
    
    # Display component-specific visualizations
    if component_name == "data_preprocessing":
        render_data_preprocessing_overview(component_data)
    elif component_name == "feature_extraction":
        render_feature_extraction_overview(component_data)
    elif component_name == "missing_data_handling":
        render_missing_data_overview(component_data)
    elif component_name == "expert_training":
        render_expert_training_overview(component_data)
    elif component_name == "gating_network":
        render_gating_network_overview(component_data)
    elif component_name == "moe_integration":
        render_moe_integration_overview(component_data)
    elif component_name == "output_generation":
        render_output_generation_overview(component_data)

def render_component_metrics(component_name: str, component_data: Dict[str, Any]):
    """
    Render metrics for a component.
    
    Args:
        component_name: Name of the component
        component_data: Data for this component
    """
    st.markdown(f"## Performance Metrics")
    
    # Display metrics if available
    if "metrics" in component_data and component_data["metrics"]:
        metrics = component_data["metrics"]
        
        # Convert metrics to a DataFrame for display
        metrics_df = pd.DataFrame(
            {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
        )
        
        # Convert values to strings to avoid serialization issues
        metrics_df["Value"] = metrics_df["Value"].apply(lambda x: str(x))
        
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        # Create visualization for numeric metrics
        numeric_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_metrics[k] = v
        
        if numeric_metrics:
            fig = px.bar(
                x=list(numeric_metrics.keys()),
                y=list(numeric_metrics.values()),
                title=f"Metrics for {component_name.replace('_', ' ').title()}",
                labels={"x": "Metric", "y": "Value"}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics available for this component.")

def render_component_data_flow(component_name: str, component_data: Dict[str, Any]):
    """
    Render data flow visualizations for a component.
    
    Args:
        component_name: Name of the component
        component_data: Data for this component
    """
    st.markdown(f"## Data Flow")
    
    # Generate sample data for demonstration
    if 'sample_data' not in st.session_state:
        # Generate sample data
        np.random.seed(42)
        n_samples = 5
        
        sample_data = pd.DataFrame({
            'Feature 1': np.random.randn(n_samples),
            'Feature 2': np.random.randn(n_samples),
            'Feature 3': np.random.randn(n_samples),
            'Feature 4': np.random.randn(n_samples)
        })
        
        st.session_state.sample_data = sample_data
    
    # Get sample data
    sample_data = st.session_state.sample_data
    
    # Display input data
    st.markdown("### Input Data Sample")
    st.dataframe(sample_data, use_container_width=True)
    
    # Display output data based on component
    st.markdown("### Output Data Sample")
    output_data = transform_sample_data(sample_data, component_name)
    st.dataframe(output_data, use_container_width=True)
    
    # Display data transformation visualization based on component
    st.markdown("### Data Transformation")
    
    if component_name == "data_preprocessing":
        render_data_preprocessing_flow(sample_data, output_data)
    elif component_name == "feature_extraction":
        render_feature_extraction_flow(sample_data, output_data)
    elif component_name == "missing_data_handling":
        render_missing_data_flow(sample_data, output_data)
    elif component_name == "expert_training":
        render_expert_training_flow(sample_data, output_data)
    elif component_name == "gating_network":
        render_gating_network_flow(sample_data, output_data)
    elif component_name == "moe_integration":
        render_moe_integration_flow(sample_data, output_data)
    elif component_name == "output_generation":
        render_output_generation_flow(sample_data, output_data)

def transform_sample_data(data: pd.DataFrame, component_name: str) -> pd.DataFrame:
    """
    Transform sample data based on the component.
    
    Args:
        data: Input sample data
        component_name: Name of the component
        
    Returns:
        Transformed sample data
    """
    if component_name == "data_preprocessing":
        # Add preprocessing indicators
        output = data.copy()
        output["Normalized"] = True
        output["Outlier_Removed"] = [False, False, True, False, False]
        return output
    
    elif component_name == "feature_extraction":
        # Replace with extracted features
        return pd.DataFrame({
            "Extracted_Feature_1": data["Feature 1"] * 0.8 + data["Feature 2"] * 0.2,
            "Extracted_Feature_2": data["Feature 3"] * 0.7 + data["Feature 4"] * 0.3,
            "Extracted_Feature_3": data["Feature 1"] * 0.1 + data["Feature 4"] * 0.9
        })
    
    elif component_name == "missing_data_handling":
        # Add missing value indicators
        output = data.copy()
        output.iloc[1, 2] = np.nan  # Introduce a missing value
        output.iloc[3, 0] = np.nan  # Introduce another missing value
        output["Missing_Values_Count"] = output.isna().sum(axis=1)
        output["Imputed"] = [False, True, False, True, False]
        return output
    
    elif component_name == "expert_training":
        # Add predictions and confidence for each expert
        return pd.DataFrame({
            "Physiological_Prediction": data["Feature 1"] * 1.2,
            "Physiological_Confidence": np.random.uniform(0.7, 0.95, len(data)),
            "Behavioral_Prediction": data["Feature 2"] * 0.8,
            "Behavioral_Confidence": np.random.uniform(0.6, 0.9, len(data)),
            "Environmental_Prediction": data["Feature 3"] * 1.1,
            "Environmental_Confidence": np.random.uniform(0.5, 0.85, len(data))
        })
    
    elif component_name == "gating_network":
        # Add expert weights
        return pd.DataFrame({
            "Physiological_Weight": np.random.uniform(0.2, 0.5, len(data)),
            "Behavioral_Weight": np.random.uniform(0.1, 0.4, len(data)),
            "Environmental_Weight": np.random.uniform(0.3, 0.6, len(data)),
            "Selected_Expert": ["Physiological", "Environmental", "Behavioral", 
                               "Environmental", "Physiological"]
        })
    
    elif component_name == "moe_integration":
        # Add integrated predictions
        expert_outputs = transform_sample_data(data, "expert_training")
        gating_weights = transform_sample_data(data, "gating_network")
        
        # Calculate weighted predictions
        weighted_pred = (
            expert_outputs["Physiological_Prediction"] * gating_weights["Physiological_Weight"] +
            expert_outputs["Behavioral_Prediction"] * gating_weights["Behavioral_Weight"] +
            expert_outputs["Environmental_Prediction"] * gating_weights["Environmental_Weight"]
        )
        
        return pd.DataFrame({
            "Integrated_Prediction": weighted_pred,
            "Ensemble_Confidence": np.random.uniform(0.8, 0.98, len(data)),
            "Physiological_Weight": gating_weights["Physiological_Weight"],
            "Behavioral_Weight": gating_weights["Behavioral_Weight"],
            "Environmental_Weight": gating_weights["Environmental_Weight"]
        })
    
    elif component_name == "output_generation":
        # Final output format
        moe_output = transform_sample_data(data, "moe_integration")
        
        return pd.DataFrame({
            "Prediction": moe_output["Integrated_Prediction"],
            "Confidence": moe_output["Ensemble_Confidence"],
            "Uncertainty": np.random.uniform(0.02, 0.2, len(data)),
            "Contributing_Experts": ["Phys, Env", "Env, Behav", "Behav", "Env, Phys", "Phys"]
        })
    
    # Default: return original data
    return data.copy()

# Component-specific overview visualizations
def render_data_preprocessing_overview(component_data: Dict[str, Any]):
    """Render data preprocessing overview visualization."""
    metrics = component_data.get("metrics", {})
    
    # Create a simple bar chart for categorical vs numerical features
    if "categorical_features" in metrics and "numerical_features" in metrics:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Categorical", "Numerical"],
            y=[metrics["categorical_features"], metrics["numerical_features"]],
            marker_color=["#FF9933", "#3366FF"]
        ))
        fig.update_layout(
            title="Feature Types",
            xaxis_title="Feature Type",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Create pie chart for data quality
    if "data_quality_score" in metrics:
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=["Good Quality", "Issues"],
            values=[metrics["data_quality_score"], 1 - metrics["data_quality_score"]],
            marker_colors=["#33CC66", "#FF6666"]
        ))
        fig.update_layout(title="Data Quality Score")
        st.plotly_chart(fig, use_container_width=True)

def render_feature_extraction_overview(component_data: Dict[str, Any]):
    """Render feature extraction overview visualization."""
    metrics = component_data.get("metrics", {})
    
    # Create a visualization for explained variance
    if "variance_explained" in metrics:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics["variance_explained"] * 100,
            title={"text": "Variance Explained"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3366FF"},
                "steps": [
                    {"range": [0, 60], "color": "#FF6666"},
                    {"range": [60, 80], "color": "#FFCC66"},
                    {"range": [80, 100], "color": "#66CC66"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Display top features
    if "top_features" in metrics and isinstance(metrics["top_features"], list):
        st.markdown("### Top Features")
        for i, feature in enumerate(metrics["top_features"]):
            st.markdown(f"{i+1}. **{feature}**")

def render_missing_data_overview(component_data: Dict[str, Any]):
    """Render missing data handling overview visualization."""
    metrics = component_data.get("metrics", {})
    
    # Create a visualization for imputation methods
    if "imputation_methods" in metrics and isinstance(metrics["imputation_methods"], list):
        methods = metrics["imputation_methods"]
        # Create random counts for each method
        np.random.seed(42)
        counts = np.random.randint(10, 100, len(methods))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=methods,
            y=counts,
            marker_color="#3366FF"
        ))
        fig.update_layout(
            title="Imputation Methods Used",
            xaxis_title="Method",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Create a visualization for data completeness
    if "data_completeness" in metrics:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics["data_completeness"] * 100,
            title={"text": "Data Completeness"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3366FF"},
                "steps": [
                    {"range": [0, 60], "color": "#FF6666"},
                    {"range": [60, 90], "color": "#FFCC66"},
                    {"range": [90, 100], "color": "#66CC66"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

def render_expert_training_overview(component_data: Dict[str, Any]):
    """Render expert training overview visualization."""
    metrics = component_data.get("metrics", {})
    experts = component_data.get("experts", {})
    
    # Create a visualization for expert training status
    if experts:
        expert_names = list(experts.keys())
        trained_status = [int(expert.get("trained", False)) for expert in experts.values()]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=expert_names,
            y=trained_status,
            marker_color=["#66CC66" if status else "#FF6666" for status in trained_status]
        ))
        fig.update_layout(
            title="Expert Training Status",
            xaxis_title="Expert",
            yaxis_title="Trained (1) / Not Trained (0)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Create a visualization for validation scores
    if "average_validation_score" in metrics:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics["average_validation_score"] * 100,
            title={"text": "Average Validation Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3366FF"},
                "steps": [
                    {"range": [0, 60], "color": "#FF6666"},
                    {"range": [60, 80], "color": "#FFCC66"},
                    {"range": [80, 100], "color": "#66CC66"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

def render_gating_network_overview(component_data: Dict[str, Any]):
    """Render gating network overview visualization."""
    metrics = component_data.get("metrics", {})
    
    # Create a visualization for routing accuracy
    if "routing_accuracy" in metrics:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics["routing_accuracy"] * 100,
            title={"text": "Routing Accuracy"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3366FF"},
                "steps": [
                    {"range": [0, 60], "color": "#FF6666"},
                    {"range": [60, 80], "color": "#FFCC66"},
                    {"range": [80, 100], "color": "#66CC66"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Create a visualization for expert usage
    if "experts_used" in metrics and "average_weight" in metrics:
        # Create a random distribution of weights
        np.random.seed(42)
        expert_count = metrics["experts_used"]
        expert_names = [f"Expert {i+1}" for i in range(expert_count)]
        weights = np.random.dirichlet(np.ones(expert_count))
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=expert_names,
            values=weights,
            marker_colors=px.colors.qualitative.Set3[:expert_count]
        ))
        fig.update_layout(title="Expert Weight Distribution")
        st.plotly_chart(fig, use_container_width=True)

def render_moe_integration_overview(component_data: Dict[str, Any]):
    """Render MoE integration overview visualization."""
    metrics = component_data.get("metrics", {})
    
    # Create a visualization for ensemble accuracy
    if "ensemble_accuracy" in metrics:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics["ensemble_accuracy"] * 100,
            title={"text": "Ensemble Accuracy"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3366FF"},
                "steps": [
                    {"range": [0, 60], "color": "#FF6666"},
                    {"range": [60, 80], "color": "#FFCC66"},
                    {"range": [80, 100], "color": "#66CC66"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Create a visualization for expert importance
    if "expert_importances" in component_data:
        importances = component_data["expert_importances"]
        expert_names = list(importances.keys())
        importance_values = list(importances.values())
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=expert_names,
            y=importance_values,
            marker_color=px.colors.qualitative.Set3[:len(expert_names)]
        ))
        fig.update_layout(
            title="Expert Importance in Ensemble",
            xaxis_title="Expert",
            yaxis_title="Importance"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif "best_expert" in metrics:
        st.markdown(f"**Best Expert:** {metrics['best_expert']}")

def render_output_generation_overview(component_data: Dict[str, Any]):
    """Render output generation overview visualization."""
    metrics = component_data.get("metrics", {})
    
    # Create a visualization for prediction confidence
    if "prediction_confidence" in metrics:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics["prediction_confidence"] * 100,
            title={"text": "Prediction Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3366FF"},
                "steps": [
                    {"range": [0, 60], "color": "#FF6666"},
                    {"range": [60, 80], "color": "#FFCC66"},
                    {"range": [80, 100], "color": "#66CC66"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Create a visualization for predictions generated
    if "predictions_generated" in metrics:
        st.markdown(f"**Predictions Generated:** {metrics['predictions_generated']}")
        
        # Create a random distribution of predictions
        np.random.seed(42)
        predictions = np.random.normal(0, 1, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=predictions,
            marker_color="#3366FF"
        ))
        fig.update_layout(
            title="Prediction Distribution",
            xaxis_title="Prediction Value",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)

# Component-specific data flow visualizations
def render_data_preprocessing_flow(input_data: pd.DataFrame, output_data: pd.DataFrame):
    """Render data preprocessing flow visualization."""
    st.info("Data preprocessing transforms raw input data by normalizing numerical features, handling outliers, and standardizing formats.")
    
    # Create a simple before/after visualization for a feature
    if "Feature 1" in input_data.columns:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=input_data["Feature 1"],
            name="Before Preprocessing",
            marker_color="#FF9933"
        ))
        fig.add_trace(go.Box(
            y=output_data["Feature 1"],
            name="After Preprocessing",
            marker_color="#3366FF"
        ))
        fig.update_layout(
            title="Feature 1 Before and After Preprocessing",
            yaxis_title="Value"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_feature_extraction_flow(input_data: pd.DataFrame, output_data: pd.DataFrame):
    """Render feature extraction flow visualization."""
    st.info("Feature extraction transforms original features into new, more informative features through dimensionality reduction and feature engineering.")
    
    # Create a visualization comparing original vs extracted features
    fig = go.Figure()
    
    # Add input data features
    for col in input_data.columns:
        fig.add_trace(go.Box(
            y=input_data[col],
            name=f"Original: {col}",
            marker_color="#FF9933"
        ))
    
    # Add output data features
    for col in output_data.columns:
        fig.add_trace(go.Box(
            y=output_data[col],
            name=f"Extracted: {col}",
            marker_color="#3366FF"
        ))
    
    fig.update_layout(
        title="Original vs Extracted Features",
        yaxis_title="Value"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_missing_data_flow(input_data: pd.DataFrame, output_data: pd.DataFrame):
    """Render missing data handling flow visualization."""
    st.info("Missing data handling identifies and replaces missing values using sophisticated imputation techniques.")
    
    # Create a visualization showing missing values
    fig = go.Figure()
    
    # Create a heatmap-like visualization
    heat_data = output_data.isna().astype(int)
    
    # Create a heatmap
    fig = px.imshow(
        heat_data,
        labels=dict(x="Features", y="Samples", color="Missing (1) / Present (0)"),
        x=heat_data.columns,
        y=[f"Sample {i+1}" for i in range(len(heat_data))],
        color_continuous_scale=["#66CC66", "#FF6666"]
    )
    
    fig.update_layout(title="Missing Value Heatmap")
    st.plotly_chart(fig, use_container_width=True)

def render_expert_training_flow(input_data: pd.DataFrame, output_data: pd.DataFrame):
    """Render expert training flow visualization."""
    st.info("Expert training involves each specialized model learning to predict the target based on its domain-specific features.")
    
    # Create a scatter plot for each expert's predictions
    fig = go.Figure()
    
    # Generate random target values for demonstration
    np.random.seed(42)
    target = np.random.normal(0, 1, len(input_data))
    
    # Plot each expert's predictions against the target
    fig.add_trace(go.Scatter(
        x=target,
        y=output_data["Physiological_Prediction"],
        mode="markers",
        name="Physiological Expert",
        marker=dict(color="#FF9933", size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=target,
        y=output_data["Behavioral_Prediction"],
        mode="markers",
        name="Behavioral Expert",
        marker=dict(color="#3366FF", size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=target,
        y=output_data["Environmental_Prediction"],
        mode="markers",
        name="Environmental Expert",
        marker=dict(color="#66CC66", size=10)
    ))
    
    # Add perfect prediction line
    fig.add_trace(go.Scatter(
        x=[-3, 3],
        y=[-3, 3],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="black", dash="dash")
    ))
    
    fig.update_layout(
        title="Expert Predictions vs. Target",
        xaxis_title="Target Value",
        yaxis_title="Predicted Value"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_gating_network_flow(input_data: pd.DataFrame, output_data: pd.DataFrame):
    """Render gating network flow visualization."""
    st.info("The gating network determines which experts to trust for each input by assigning weights to each expert's prediction.")
    
    # Create a stacked bar chart for expert weights
    fig = go.Figure()
    
    for i in range(len(output_data)):
        fig.add_trace(go.Bar(
            x=[f"Sample {i+1}"],
            y=[output_data["Physiological_Weight"].iloc[i]],
            name="Physiological" if i == 0 else None,
            marker_color="#FF9933",
            showlegend=(i == 0)
        ))
        
        fig.add_trace(go.Bar(
            x=[f"Sample {i+1}"],
            y=[output_data["Behavioral_Weight"].iloc[i]],
            name="Behavioral" if i == 0 else None,
            marker_color="#3366FF",
            showlegend=(i == 0)
        ))
        
        fig.add_trace(go.Bar(
            x=[f"Sample {i+1}"],
            y=[output_data["Environmental_Weight"].iloc[i]],
            name="Environmental" if i == 0 else None,
            marker_color="#66CC66",
            showlegend=(i == 0)
        ))
    
    fig.update_layout(
        title="Expert Weights Assigned by Gating Network",
        xaxis_title="Sample",
        yaxis_title="Weight",
        barmode="stack"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_moe_integration_flow(input_data: pd.DataFrame, output_data: pd.DataFrame):
    """Render MoE integration flow visualization."""
    st.info("MoE integration combines predictions from multiple experts using weights from the gating network to produce a unified prediction.")
    
    # Create a bar chart comparing integrated predictions with individual expert contributions
    fig = go.Figure()
    
    # Get expert predictions
    expert_data = transform_sample_data(input_data, "expert_training")
    
    for i in range(len(output_data)):
        # Expert contributions
        physiological_contrib = output_data["Physiological_Weight"].iloc[i] * expert_data["Physiological_Prediction"].iloc[i]
        behavioral_contrib = output_data["Behavioral_Weight"].iloc[i] * expert_data["Behavioral_Prediction"].iloc[i]
        environmental_contrib = output_data["Environmental_Weight"].iloc[i] * expert_data["Environmental_Prediction"].iloc[i]
        
        fig.add_trace(go.Bar(
            x=[f"Sample {i+1}"],
            y=[physiological_contrib],
            name="Physiological Contribution" if i == 0 else None,
            marker_color="#FF9933",
            showlegend=(i == 0)
        ))
        
        fig.add_trace(go.Bar(
            x=[f"Sample {i+1}"],
            y=[behavioral_contrib],
            name="Behavioral Contribution" if i == 0 else None,
            marker_color="#3366FF",
            showlegend=(i == 0)
        ))
        
        fig.add_trace(go.Bar(
            x=[f"Sample {i+1}"],
            y=[environmental_contrib],
            name="Environmental Contribution" if i == 0 else None,
            marker_color="#66CC66",
            showlegend=(i == 0)
        ))
        
        # Integrated prediction (line)
        fig.add_trace(go.Scatter(
            x=[f"Sample {i+1}"],
            y=[output_data["Integrated_Prediction"].iloc[i]],
            mode="markers",
            marker=dict(color="black", size=10),
            name="Integrated Prediction" if i == 0 else None,
            showlegend=(i == 0)
        ))
    
    fig.update_layout(
        title="Expert Contributions to Integrated Prediction",
        xaxis_title="Sample",
        yaxis_title="Contribution",
        barmode="stack"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_output_generation_flow(input_data: pd.DataFrame, output_data: pd.DataFrame):
    """Render output generation flow visualization."""
    st.info("Output generation formats the final predictions and adds metadata like confidence intervals and uncertainty estimates.")
    
    # Create a visualization with predictions and confidence intervals
    fig = go.Figure()
    
    for i in range(len(output_data)):
        prediction = output_data["Prediction"].iloc[i]
        uncertainty = output_data["Uncertainty"].iloc[i]
        
        # Add prediction point
        fig.add_trace(go.Scatter(
            x=[f"Sample {i+1}"],
            y=[prediction],
            mode="markers",
            marker=dict(color="#3366FF", size=10),
            name="Prediction" if i == 0 else None,
            showlegend=(i == 0)
        ))
        
        # Add uncertainty range
        fig.add_trace(go.Scatter(
            x=[f"Sample {i+1}", f"Sample {i+1}"],
            y=[prediction - uncertainty, prediction + uncertainty],
            mode="lines",
            line=dict(color="#FF9933", width=2),
            name="Uncertainty Range" if i == 0 else None,
            showlegend=(i == 0)
        ))
    
    fig.update_layout(
        title="Final Predictions with Uncertainty",
        xaxis_title="Sample",
        yaxis_title="Prediction",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display contributing experts
    if "Contributing_Experts" in output_data.columns:
        st.markdown("### Contributing Experts")
        for i, experts in enumerate(output_data["Contributing_Experts"]):
            st.markdown(f"**Sample {i+1}:** {experts}")

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