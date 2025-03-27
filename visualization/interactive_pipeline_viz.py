"""
Interactive Pipeline Architecture Visualization Module.

This module provides functions for creating an interactive pipeline architecture 
visualization where each component is clickable and leads to detailed views.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
import matplotlib.pyplot as plt
import io
import time

# Try to import from local modules, with fallbacks for imports
try:
    from visualization.architecture_flow import create_sankey_diagram
    from visualization.data_utils import load_component_data, load_or_generate_input_data, run_complete_pipeline
except ImportError:
    try:
        from architecture_flow import create_sankey_diagram
    except ImportError:
        # Define a simplified version if module is not available
        def create_sankey_diagram(*args, **kwargs):
            """Fallback implementation if architecture_flow is not available."""
            return None

# Constants for component names and their IDs
PIPELINE_COMPONENTS = {
    "data_preprocessing": 0,
    "feature_extraction": 1,
    "missing_data_handling": 2,
    "expert_training": 3,
    "gating_network": 4,
    "moe_integration": 5,
    "output_generation": 6
}

# Component colors for visualization
COMPONENT_COLORS = {
    "data_preprocessing": "#1f77b4",      # Blue
    "feature_extraction": "#ff7f0e",      # Orange
    "missing_data_handling": "#2ca02c",   # Green
    "expert_training": "#d62728",         # Red
    "gating_network": "#9467bd",          # Purple
    "moe_integration": "#8c564b",         # Brown
    "output_generation": "#e377c2"        # Pink
}

# Import data utilities and pipeline visualization functions
from visualization.data_utils import load_or_generate_input_data, run_complete_pipeline
from visualization.component_details import render_component_details

# Try to import improved MoE integration first, with fallback to original
improved_integration_available = False
original_integration_available = False

try:
    from visualization.improved_moe_integration import run_moe_pipeline as improved_run_pipeline, MOE_MODULES_AVAILABLE as IMPROVED_AVAILABLE
    improved_integration_available = IMPROVED_AVAILABLE
    if improved_integration_available:
        st.sidebar.success("✅ Using improved MoE integration")
except ImportError:
    st.sidebar.warning("⚠️ Improved MoE integration not available")
    improved_integration_available = False

# Try original integration if improved is not available
if not improved_integration_available:
    try:
        from visualization.moe_integration import run_moe_pipeline as original_run_pipeline, MOE_MODULES_AVAILABLE as ORIGINAL_AVAILABLE
        original_integration_available = ORIGINAL_AVAILABLE
        if original_integration_available:
            st.sidebar.success("✅ Using original MoE integration")
    except ImportError:
        st.sidebar.warning("⚠️ Original MoE integration not available")
        original_integration_available = False

def create_interactive_pipeline():
    """Create an interactive visualization of the MoE pipeline architecture."""
    st.markdown("## MoE Pipeline Architecture")
    st.markdown("Click on any component to view detailed metrics and visualizations")
    
    # Define the nodes
    nodes = [
        ("data_preprocessing", "Data Preprocessing"),
        ("feature_extraction", "Feature Extraction"),
        ("missing_data_handling", "Missing Data Handling"),
        ("expert_training", "Expert Training"),
        ("gating_network", "Gating Network"),
        ("moe_integration", "MoE Integration"),
        ("output_generation", "Output Generation")
    ]
    
    # Define the edges
    edges = [
        ("data_preprocessing", "feature_extraction"),
        ("feature_extraction", "missing_data_handling"),
        ("missing_data_handling", "expert_training"),
        ("missing_data_handling", "gating_network"),
        ("expert_training", "moe_integration"),
        ("gating_network", "moe_integration"),
        ("moe_integration", "output_generation")
    ]
    
    # Create the graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node_id, node_label in nodes:
        G.add_node(node_id, label=node_label)
    
    # Add edges
    G.add_edges_from(edges)
    
    # Define node positions
    pos = {
        "data_preprocessing": (0, 0),
        "feature_extraction": (1, 0),
        "missing_data_handling": (2, 0),
        "expert_training": (3, 1),
        "gating_network": (3, -1),
        "moe_integration": (4, 0),
        "output_generation": (5, 0)
    }
    
    # Get the currently selected component from session state
    selected_component = st.session_state.get("selected_component", None)
    
    # Create the figure
    plt.figure(figsize=(12, 6))
    
    # Node colors based on selection
    node_colors = []
    for node in G.nodes():
        if node == selected_component:
            node_colors.append("#FF7F0E")  # Highlighted orange for selected
        else:
            node_colors.append("#1F77B4")  # Default blue
    
    # Edge colors - highlight edges connected to selected node
    edge_colors = []
    for u, v in G.edges():
        if u == selected_component or v == selected_component:
            edge_colors.append('#FF7F0E')  # Highlighted orange
        else:
            edge_colors.append('#AAAAAA')  # Default gray
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20, edge_color=edge_colors)
    nx.draw_networkx_labels(G, pos, {n: G.nodes[n]['label'] for n in G.nodes()}, font_size=10, font_weight='bold')
    
    # Turn off the axis
    plt.axis('off')
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Display the figure
    st.image(buf, use_container_width=True)
    
    # Create clickable buttons for each component with tooltips
    st.markdown("### Select a component to explore")
    
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7 = st.columns(3)
    
    # For hover tooltips and more descriptive button text
    component_descriptions = {
        "data_preprocessing": "Handles data cleaning, normalization, and initial processing",
        "feature_extraction": "Extracts meaningful features from raw data",
        "missing_data_handling": "Handles missing values with imputation techniques",
        "expert_training": "Trains specialized expert models",
        "gating_network": "Routes inputs to appropriate experts",
        "moe_integration": "Combines expert outputs based on gating weights",
        "output_generation": "Produces final predictions and results"
    }
    
    # Button styling based on selection
    def get_button_style(component_id):
        if component_id == selected_component:
            return "primary"
        return "secondary"
    
    # Row 1
    with col1:
        st.button(
            f"📊 {G.nodes['data_preprocessing']['label']}", 
            key="btn_data_preprocessing",
            help=component_descriptions["data_preprocessing"],
            type=get_button_style("data_preprocessing"),
            on_click=select_component, args=["data_preprocessing"]
        )
    
    with col2:
        st.button(
            f"🔍 {G.nodes['feature_extraction']['label']}",
            key="btn_feature_extraction",
            help=component_descriptions["feature_extraction"],
            type=get_button_style("feature_extraction"),
            on_click=select_component, args=["feature_extraction"]
        )
    
    with col3:
        st.button(
            f"🧩 {G.nodes['missing_data_handling']['label']}",
            key="btn_missing_data_handling",
            help=component_descriptions["missing_data_handling"],
            type=get_button_style("missing_data_handling"),
            on_click=select_component, args=["missing_data_handling"]
        )
    
    with col4:
        st.button(
            f"🧠 {G.nodes['expert_training']['label']}",
            key="btn_expert_training",
            help=component_descriptions["expert_training"],
            type=get_button_style("expert_training"),
            on_click=select_component, args=["expert_training"]
        )
    
    # Row 2
    with col5:
        st.button(
            f"🔀 {G.nodes['gating_network']['label']}",
            key="btn_gating_network",
            help=component_descriptions["gating_network"],
            type=get_button_style("gating_network"),
            on_click=select_component, args=["gating_network"]
        )
    
    with col6:
        st.button(
            f"🔗 {G.nodes['moe_integration']['label']}",
            key="btn_moe_integration",
            help=component_descriptions["moe_integration"],
            type=get_button_style("moe_integration"),
            on_click=select_component, args=["moe_integration"]
        )
    
    with col7:
        st.button(
            f"📈 {G.nodes['output_generation']['label']}",
            key="btn_output_generation",
            help=component_descriptions["output_generation"],
            type=get_button_style("output_generation"),
            on_click=select_component, args=["output_generation"]
        )
    
    # If a component is selected, display the details
    if selected_component:
        st.markdown(f"### {G.nodes[selected_component]['label']} Details")
        st.markdown(f"*{component_descriptions[selected_component]}*")
        st.divider()
        render_component_details(selected_component)

def add_interactive_pipeline_architecture():
    """
    Add the interactive pipeline architecture visualization to the Streamlit app.
    Handles component selection and detail views.
    """
    # Initialize rerun flag if it doesn't exist
    if 'needs_rerun' not in st.session_state:
        st.session_state.needs_rerun = False
        
    st.subheader("MoE Pipeline Architecture")
    
    # Add pipeline control section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        👆 Click on any component in the pipeline to see detailed information and metrics.
        
        This visualization shows how data flows through your Mixture of Experts pipeline.
        Each component processes data and passes it to the next stage.
        """)
    
    with col2:
        # Check if we're using real pipeline data
        using_pipeline_data = False
        if 'pipeline_data' in st.session_state and st.session_state.pipeline_data:
            using_pipeline_data = True
            st.success("✅ Using real MoE pipeline data")
        else:
            st.warning("⚠️ Using sample visualization data")
        
        # Add button to run pipeline
        if st.button("Run MoE Pipeline"):
            # Check if we are using real data
            using_real_data = st.session_state.get('use_real_data', False)
            
            # Debug information
            st.info("🔄 Starting MoE pipeline execution...")
            st.write("Debug: Checking integration availability")
            st.write(f"Improved integration available: {improved_integration_available}")
            st.write(f"Original integration available: {original_integration_available}")
            
            with st.spinner("🔄 Running MoE pipeline..."):
                # Try to get input data, or generate synthetic data
                st.write("Debug: Loading or generating input data")
                input_data = load_or_generate_input_data(synthetic=not using_real_data)
                
                if input_data is not None:
                    st.write(f"Debug: Input data shape: {input_data.shape if hasattr(input_data, 'shape') else 'unknown'}")
                else:
                    st.error("Debug: Input data is None!")
                
                # Always use the data_utils pipeline for consistency
                st.write("Debug: Using data_utils pipeline implementation")
                try:
                    from visualization.data_utils import run_complete_pipeline
                    results, pipeline_id = run_complete_pipeline(input_data)
                    st.write(f"Debug: Pipeline execution completed, pipeline_id: {pipeline_id}")
                    if isinstance(results, dict):
                        st.write(f"Debug: Results keys: {list(results.keys())}")
                    else:
                        st.write(f"Debug: Results is not a dict, type: {type(results)}")
                    
                    st.session_state['input_data'] = input_data
                    st.session_state['pipeline_data'] = results
                    st.session_state['pipeline_id'] = pipeline_id
                    st.success(f"✅ Pipeline executed successfully! Pipeline ID: {pipeline_id}")
                except Exception as e:
                    st.error(f"❌ Error running MoE pipeline: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
    
    # Initialize session state for keeping track of selected component
    if 'selected_component' not in st.session_state:
        st.session_state.selected_component = None
    
    # Create and display the interactive pipeline
    create_interactive_pipeline()
    
    # Check if we need to rerun (after rendering everything)
    if st.session_state.needs_rerun:
        st.session_state.needs_rerun = False
        st.rerun()

def select_component(component_name: str):
    """
    Handle component selection and update session state.
    
    Args:
        component_name: Name of the component to select
    """
    # Store the selected component in session state
    st.session_state.selected_component = component_name
    
    # If we're using real pipeline data, make sure to load the component data
    if 'pipeline_data' in st.session_state and st.session_state.pipeline_data:
        # Check if we need to run the pipeline up to this component
        if component_name not in st.session_state.pipeline_data:
            # Import modules
            from visualization.data_utils import process_data_through_pipeline
            
            with st.spinner(f"Processing data through pipeline up to {component_name}..."):
                # Get input data
                input_data = st.session_state.input_data
                
                # Process data up to the selected component
                results = process_data_through_pipeline(input_data, up_to_component=component_name)
                
                # Update session state
                st.session_state.pipeline_data.update(results)
    
    # Set a flag to rerun on the next render pass instead of calling rerun directly
    st.session_state.needs_rerun = True

def render_component_details(component_name: str):
    """
    Render detailed information for a specific pipeline component.
    
    Args:
        component_name: The name of the component to display details for
    """
    # Create a container for the component details
    st.markdown("---")
    st.subheader(f"{component_name.replace('_', ' ').title()} Details")
    
    # Add a back button to return to the overview
    if st.button("← Back to Pipeline Overview"):
        st.session_state.selected_component = None
        st.session_state.needs_rerun = True
    
    # Create tabs for different aspects of the component
    overview_tab, metrics_tab, data_tab = st.tabs(["Overview", "Metrics", "Data Samples"])
    
    # Get component ID for reference
    component_id = PIPELINE_COMPONENTS.get(component_name, 0)
    
    # Render content based on the selected component
    with overview_tab:
        render_component_overview(component_name)
    
    with metrics_tab:
        render_component_metrics(component_name)
    
    with data_tab:
        render_component_data_samples(component_name)

def render_component_overview(component_name: str):
    """Render overview information for a component."""
    # Define overview content for each component
    overviews = {
        "data_preprocessing": {
            "description": "The Data Preprocessing component cleans and transforms raw input data into a suitable format for feature extraction and model training.",
            "key_functions": ["Data cleaning", "Normalization", "Encoding categorical variables", "Handling missing values"],
            "image_path": None
        },
        "feature_extraction": {
            "description": "The Feature Extraction component identifies and extracts relevant features from preprocessed data to be used by expert models.",
            "key_functions": ["Feature selection", "Dimensionality reduction", "Feature engineering", "Feature scaling"],
            "image_path": None
        },
        "missing_data_handling": {
            "description": "The Missing Data Handling component applies specialized imputation techniques to handle missing values in the dataset.",
            "key_functions": ["Pattern recognition", "Imputation models", "Missing value prediction", "Quality assessment"],
            "image_path": None
        },
        "expert_training": {
            "description": "The Expert Training component trains specialized models (experts) on different aspects or subsets of the data.",
            "key_functions": ["Model selection", "Hyperparameter tuning", "Training loops", "Validation"],
            "image_path": None
        },
        "gating_network": {
            "description": "The Gating Network determines which expert models should handle each input example and with what weight.",
            "key_functions": ["Expert selection", "Confidence estimation", "Input space partitioning", "Weight assignment"],
            "image_path": None
        },
        "moe_integration": {
            "description": "The MoE Integration component combines outputs from multiple experts based on gating network weights.",
            "key_functions": ["Weighted averaging", "Ensemble techniques", "Confidence calibration", "Output normalization"],
            "image_path": None
        },
        "output_generation": {
            "description": "The Output Generation component produces final predictions and explanations based on the integrated expert outputs.",
            "key_functions": ["Prediction formatting", "Uncertainty estimation", "Explainability", "Result evaluation"],
            "image_path": None
        }
    }
    
    # Get overview for the selected component
    overview = overviews.get(component_name, {
        "description": "No description available for this component.",
        "key_functions": [],
        "image_path": None
    })
    
    # Display overview information
    st.markdown(f"### {component_name.replace('_', ' ').title()}")
    st.markdown(overview["description"])
    
    # Display key functions
    st.markdown("#### Key Functions")
    for func in overview["key_functions"]:
        st.markdown(f"- {func}")
    
    # Display image if available
    if overview["image_path"]:
        st.image(overview["image_path"], use_container_width=True)
    
    # Add placeholder for workflow selection
    st.markdown("#### Select Workflow")
    workflow_id = st.selectbox(
        "Choose a workflow to visualize data for this component:",
        ["latest_workflow", "workflow_123", "workflow_456"],
        key=f"{component_name}_workflow_select"
    )

def render_component_metrics(component_name: str):
    """Render metrics for a component."""
    st.markdown("### Performance Metrics")
    
    # First check if we have real pipeline data in session state
    real_metrics_available = False
    if 'pipeline_data' in st.session_state and st.session_state.pipeline_data:
        if component_name in st.session_state.pipeline_data:
            component_data = st.session_state.pipeline_data[component_name]
            if "metrics" in component_data and component_data["metrics"]:
                real_metrics_available = True
                
                # Display actual metrics from pipeline
                metrics = component_data["metrics"]
                if isinstance(metrics, dict) and metrics:
                    # Convert all values to strings to prevent PyArrow conversion issues
                    metrics_str = {k: str(v) for k, v in metrics.items()}
                    
                    # Convert metrics to DataFrame for display
                    metrics_df = pd.DataFrame({
                        "Metric": list(metrics_str.keys()),
                        "Value": list(metrics_str.values())
                    })
                    st.dataframe(metrics_df, hide_index=True)
                    
                    # Create a visualization for numeric metrics
                    numeric_metrics = {k: v for k, v in metrics.items() 
                                    if isinstance(v, (int, float)) and not isinstance(v, bool)}
                    
                    if numeric_metrics:
                        fig = px.bar(
                            x=list(numeric_metrics.keys()),
                            y=list(numeric_metrics.values()),
                            labels={"x": "Metric", "y": "Value"},
                            title=f"Metrics for {component_name.replace('_', ' ').title()}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No metrics data available for this component.")
            else:
                st.info("No metrics data available for this component.")
    
    if not real_metrics_available:
        st.info("No real pipeline metrics available. Run the pipeline to see actual metrics.")

def render_component_data_samples(component_name: str):
    """Render data samples for a component."""
    st.markdown("### Input/Output Data Samples")
    
    # Generate generic sample data
    sample_data = pd.DataFrame(
        np.random.randn(5, 4),
        columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
    )
    
    # Display input data
    st.markdown("#### Input Data")
    st.dataframe(sample_data)
    
    # Display output data (modified based on component)
    st.markdown("#### Output Data")
    
    # Modify sample data based on the component
    output_data = sample_data.copy()
    
    if component_name == "data_preprocessing":
        # Add preprocessing indicators
        output_data["Normalized"] = True
        output_data["Outlier"] = [False, False, True, False, False]
    
    elif component_name == "feature_extraction":
        # Replace with extracted features
        output_data = pd.DataFrame(
            np.random.randn(5, 3),
            columns=["Extracted Feature 1", "Extracted Feature 2", "Extracted Feature 3"]
        )
    
    elif component_name == "expert_training":
        # Add predictions and confidence
        output_data["Expert 1 Prediction"] = np.random.randn(5)
        output_data["Expert 1 Confidence"] = np.random.uniform(0.7, 0.95, 5)
    
    elif component_name == "gating_network":
        # Add expert weights
        output_data["Expert 1 Weight"] = np.random.uniform(0.1, 0.4, 5)
        output_data["Expert 2 Weight"] = np.random.uniform(0.3, 0.6, 5)
        output_data["Expert 3 Weight"] = np.random.uniform(0.1, 0.3, 5)
    
    elif component_name == "moe_integration":
        # Add integrated predictions
        output_data["Integrated Prediction"] = np.random.randn(5)
        output_data["Confidence"] = np.random.uniform(0.8, 0.98, 5)
    
    elif component_name == "output_generation":
        # Final output format
        output_data = pd.DataFrame({
            "Prediction": np.random.randn(5),
            "Confidence": np.random.uniform(0.8, 0.98, 5),
            "Uncertainty": np.random.uniform(0.02, 0.2, 5),
            "Contributing Experts": ["E1, E3", "E2", "E1, E2, E3", "E3", "E1, E2"]
        })
    
    st.dataframe(output_data)
    
    # Add download buttons for the data
    st.download_button(
        label="Download Input Data CSV",
        data=sample_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{component_name}_input_data.csv",
        mime="text/csv"
    )
    
    st.download_button(
        label="Download Output Data CSV",
        data=output_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{component_name}_output_data.csv",
        mime="text/csv"
    )

def draw_pipeline_graph(stage_data):
    """
    Draw a graph of the pipeline components and their relationships.
    
    Args:
        stage_data: Dictionary of pipeline stage data
    """
    # Create nodes for each pipeline component
    nodes = []
    edges = []
    
    # Node positions
    positions = {
        "data_preprocessing": (1, 5),
        "feature_extraction": (2, 5),
        "missing_data_handling": (3, 5),
        "expert_training": (4, 5),
        "gating_network": (4, 3),
        "moe_integration": (5, 5),
        "output_generation": (6, 5)
    }
    
    # Node colors and sizes
    node_color = {
        "data_preprocessing": "lightblue",
        "feature_extraction": "lightgreen",
        "missing_data_handling": "peachpuff",
        "expert_training": "lavender",
        "gating_network": "lightcoral",
        "moe_integration": "lightyellow",
        "output_generation": "lightcyan"
    }
    
    # Create nodes
    for component_id, (x, y) in positions.items():
        if component_id in stage_data:
            color = node_color.get(component_id, "lightgrey")
            nodes.append(
                go.Scatter(
                    x=[x], 
                    y=[y], 
                    mode="markers+text",
                    marker=dict(size=40, color=color, line=dict(width=2, color="black")),
                    text=[component_id.replace("_", "<br>")],
                    name=component_id,
                    textposition="middle center",
                    hoverinfo="name"
                )
            )
    
    # Create edges
    edge_list = [
        ("data_preprocessing", "feature_extraction"),
        ("feature_extraction", "missing_data_handling"),
        ("missing_data_handling", "expert_training"),
        ("expert_training", "moe_integration"),
        ("gating_network", "moe_integration"),
        ("moe_integration", "output_generation")
    ]
    
    for start, end in edge_list:
        if start in positions and end in positions:
            start_x, start_y = positions[start]
            end_x, end_y = positions[end]
            
            # Draw arrow
            edges.append(
                go.Scatter(
                    x=[start_x, end_x],
                    y=[start_y, end_y],
                    mode="lines",
                    line=dict(width=2, color="black"),
                    showlegend=False,
                    hoverinfo="none"
                )
            )
    
    # Create figure
    fig = go.Figure(nodes + edges)
    
    # Update layout
    fig.update_layout(
        title="MoE Pipeline Components",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        plot_bgcolor='white'
    )
    
    return fig

def render_component_button(component_name, icon="🔍"):
    """
    Render a button for the component.
    
    Args:
        component_name: Name of the component
        icon: Icon to display
    
    Returns:
        True if button is clicked, False otherwise
    """
    label = component_name.replace("_", " ").title()
    return st.button(f"{icon} {label}", key=f"btn_{component_name}")

def render_pipeline_visualization():
    """
    Render the interactive pipeline visualization.
    """
    st.header("MoE Pipeline Visualization")
    
    # Get pipeline data from session state
    pipeline_data = st.session_state.get('pipeline_data', {})
    
    # Button to run the MoE pipeline
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("Run MoE Pipeline", key="run_moe_pipeline"):
            # Check if we are using real data
            using_real_data = st.session_state.get('use_real_data', False)
            
            # Debug information
            st.info("🔄 Starting MoE pipeline execution...")
            st.write("Debug: Checking integration availability")
            st.write(f"Improved integration available: {improved_integration_available}")
            st.write(f"Original integration available: {original_integration_available}")
            
            with st.spinner("🔄 Running MoE pipeline..."):
                # Try to get input data, or generate synthetic data
                st.write("Debug: Loading or generating input data")
                input_data = load_or_generate_input_data(synthetic=not using_real_data)
                
                if input_data is not None:
                    st.write(f"Debug: Input data shape: {input_data.shape if hasattr(input_data, 'shape') else 'unknown'}")
                else:
                    st.error("Debug: Input data is None!")
                
                # Always use the data_utils pipeline for consistency
                st.write("Debug: Using data_utils pipeline implementation")
                try:
                    from visualization.data_utils import run_complete_pipeline
                    results, pipeline_id = run_complete_pipeline(input_data)
                    st.write(f"Debug: Pipeline execution completed, pipeline_id: {pipeline_id}")
                    if isinstance(results, dict):
                        st.write(f"Debug: Results keys: {list(results.keys())}")
                    else:
                        st.write(f"Debug: Results is not a dict, type: {type(results)}")
                    
                    st.session_state['input_data'] = input_data
                    st.session_state['pipeline_data'] = results
                    st.session_state['pipeline_id'] = pipeline_id
                    st.success(f"✅ Pipeline executed successfully! Pipeline ID: {pipeline_id}")
                except Exception as e:
                    st.error(f"❌ Error running MoE pipeline: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
    
    # Draw the pipeline graph if we have data
    if pipeline_data:
        st.plotly_chart(draw_pipeline_graph(pipeline_data))
        
        # Display component details
        if "selected_component" not in st.session_state:
            st.session_state["selected_component"] = None
        
        # Create buttons for each component
        st.write("### Select a component to view details")
        
        # Create multiple columns for component buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if render_component_button("data_preprocessing", "🔄"):
                st.session_state["selected_component"] = "data_preprocessing"
                
            if render_component_button("expert_training", "👨‍🔬"):
                st.session_state["selected_component"] = "expert_training"
        
        with col2:
            if render_component_button("feature_extraction", "🧩"):
                st.session_state["selected_component"] = "feature_extraction"
                
            if render_component_button("gating_network", "🔀"):
                st.session_state["selected_component"] = "gating_network"
        
        with col3:
            if render_component_button("missing_data_handling", "🧹"):
                st.session_state["selected_component"] = "missing_data_handling"
                
            if render_component_button("moe_integration", "🔄"):
                st.session_state["selected_component"] = "moe_integration"
        
        with col4:
            if render_component_button("output_generation", "📊"):
                st.session_state["selected_component"] = "output_generation"
        
        # Display selected component details
        st.write("---")
        selected_component = st.session_state["selected_component"]
        
        if selected_component:
            component_data = pipeline_data.get(selected_component, {})
            render_component_details(selected_component, component_data)
        else:
            st.info("👆 Select a component to view its details")
    else:
        # Display placeholder if no pipeline has been run
        st.info("👆 Click the button above to run the MoE pipeline and visualize the results")
    
    # Draw expert importances if available
    if pipeline_data and "moe_integration" in pipeline_data:
        moe_data = pipeline_data["moe_integration"]
        if "expert_importances" in moe_data:
            st.write("### Expert Importances")
            importances = moe_data["expert_importances"]
            
            # Convert to DataFrame for easier plotting
            importance_df = pd.DataFrame({
                "Expert": list(importances.keys()),
                "Importance": list(importances.values())
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values("Importance", ascending=False)
            
            # Create bar chart
            fig = px.bar(
                importance_df, 
                x="Expert", 
                y="Importance",
                color="Importance",
                title="Expert Importances in the MoE Pipeline",
                color_continuous_scale="Viridis"
            )
            
            st.plotly_chart(fig)

if __name__ == "__main__":
    # For testing this module independently
    st.set_page_config(layout="wide", page_title="Interactive Pipeline Visualization")
    render_pipeline_visualization() 