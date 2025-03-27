"""
Interactive Pipeline Visualization module for MOE Framework.

This module provides visualization tools for the pipeline architecture,
allowing users to interactively explore the MOE system components.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Import visualization utilities
from visualization.component_details import render_component_details
from visualization.data_utils import process_data_through_pipeline, run_complete_pipeline

# Set up logging
logger = logging.getLogger(__name__)

# Define pipeline components and their connections
PIPELINE_COMPONENTS = [
    "data_preprocessing",
    "feature_extraction", 
    "missing_data_handling",
    "expert_training",
    "gating_network",
    "moe_integration",
    "output_generation"
]

# Component descriptions for tooltips
COMPONENT_DESCRIPTIONS = {
    "data_preprocessing": "Cleans and transforms input data",
    "feature_extraction": "Extracts relevant features from data",
    "missing_data_handling": "Handles missing values in the dataset",
    "expert_training": "Trains specialized expert models",
    "gating_network": "Routes inputs to appropriate experts",
    "moe_integration": "Combines outputs from multiple experts",
    "output_generation": "Formats final predictions and explanations"
}

# Component colors for visualization
COMPONENT_COLORS = {
    "data_preprocessing": "#3366FF",
    "feature_extraction": "#FF9933", 
    "missing_data_handling": "#66CC66",
    "expert_training": "#FF6666",
    "gating_network": "#9966FF",
    "moe_integration": "#FFCC33",
    "output_generation": "#33CCCC"
}

def create_interactive_pipeline_view():
    """
    Creates and renders the interactive pipeline visualization in the Streamlit app.
    
    This function:
    1. Creates a visual representation of the pipeline
    2. Allows users to click on components to get more details
    3. Shows component stats and execution status
    4. Provides a way to run the pipeline step-by-step
    """
    st.markdown("## Interactive Pipeline Architecture")
    
    # Check if we need to load or create a pipeline session
    if 'pipeline_id' not in st.session_state:
        st.session_state.pipeline_id = None
        st.session_state.current_component = None
        st.session_state.executed_components = []
    
    # Create row to organize buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Reset Pipeline"):
            st.session_state.pipeline_id = None
            st.session_state.current_component = None
            st.session_state.executed_components = []
            st.rerun()

    with col2:
        if st.button("▶️ Run Complete Pipeline"):
            # Only run if not already executed
            if not st.session_state.pipeline_id or len(st.session_state.executed_components) < len(PIPELINE_COMPONENTS):
                with st.spinner("Running complete pipeline..."):
                    # Get input data (from session or generate)
                    input_data = st.session_state.get('input_data', None)
                    if input_data is None:
                        st.warning("No input data available. Using synthetic data.")
                        from visualization.data_utils import load_or_generate_input_data
                        input_data = load_or_generate_input_data(synthetic=True)
                        st.session_state.input_data = input_data
                    
                    # Run the complete pipeline
                    results = run_complete_pipeline(input_data)
                    
                    # Update session state
                    st.session_state.pipeline_id = results.get('pipeline_id')
                    st.session_state.executed_components = PIPELINE_COMPONENTS.copy()
                    
                    # Select the last component by default
                    st.session_state.current_component = PIPELINE_COMPONENTS[-1]
                    
                    st.success("Pipeline executed successfully!")
                    st.rerun()

    with col3:
        # Next step button - enabled only if we haven't run all components
        if st.session_state.executed_components and len(st.session_state.executed_components) < len(PIPELINE_COMPONENTS):
            next_component = PIPELINE_COMPONENTS[len(st.session_state.executed_components)]
            if st.button(f"⏩ Next: {next_component.replace('_', ' ').title()}"):
                run_next_component()
    
    # Create the interactive pipeline diagram
    fig = create_pipeline_diagram()
    
    # Display the diagram
    pipeline_chart = st.plotly_chart(fig, use_container_width=True)

    # If component selected, show details
    if st.session_state.current_component:
        with st.expander(f"{st.session_state.current_component.replace('_', ' ').title()} Details", expanded=True):
            render_component_details(st.session_state.current_component)

def create_pipeline_diagram():
    """
    Creates the interactive pipeline diagram using Plotly.
    
    Returns:
        Plotly figure object with the pipeline visualization
    """
    # Create visualization with Plotly
    fig = go.Figure()
    
    # Calculate node positions
    num_components = len(PIPELINE_COMPONENTS)
    x_positions = np.linspace(0, 1, num_components)
    y_positions = np.ones(num_components) * 0.5
    
    # Add connections between nodes (the lines connecting components)
    for i in range(num_components - 1):
        # Get component names
        from_component = PIPELINE_COMPONENTS[i]
        to_component = PIPELINE_COMPONENTS[i+1]
        
        # Determine if this connection has been executed
        executed = (
            from_component in st.session_state.executed_components and 
            to_component in st.session_state.executed_components
        )
        
        # Set line properties based on execution status
        line_color = "#00AA00" if executed else "#AAAAAA"
        line_width = 3 if executed else 1.5
        line_dash = None if executed else "dot"
        
        # Add the connection line
        fig.add_trace(go.Scatter(
            x=[x_positions[i], x_positions[i+1]],
            y=[y_positions[i], y_positions[i+1]],
            mode="lines",
            line=dict(color=line_color, width=line_width, dash=line_dash),
            hoverinfo="none",
            showlegend=False
        ))
    
    # Add nodes for each component
    for i, component in enumerate(PIPELINE_COMPONENTS):
        # Determine if this component has been executed
        executed = component in st.session_state.executed_components
        # Determine if this is the currently selected component
        selected = st.session_state.current_component == component
        
        # Set marker properties based on state
        marker_size = 20 if selected else 16
        marker_line_width = 2 if selected else 1
        marker_line_color = "black" if selected else "#888888"
        marker_color = COMPONENT_COLORS.get(component, "#888888") if executed else "#DDDDDD"
        
        # Add the node
        fig.add_trace(go.Scatter(
            x=[x_positions[i]],
            y=[y_positions[i]],
            mode="markers+text",
            marker=dict(
                size=marker_size,
                color=marker_color,
                line=dict(width=marker_line_width, color=marker_line_color),
            ),
            text=component.replace("_", "<br>"),
            textposition="bottom center",
            name=component.replace("_", " ").title(),
            hovertext=COMPONENT_DESCRIPTIONS.get(component, ""),
            hoverinfo="text",
            customdata=[component]
        ))
    
    # Set up layout
    fig.update_layout(
        showlegend=False,
        clickmode='event+select',
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.1, 1.1]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 1]
        ),
        margin=dict(l=20, r=20, t=20, b=80),
        height=250
    )
    
    # Add click event callback
    fig.update_layout(
        annotations=[
            dict(
                x=x_positions[i],
                y=y_positions[i] - 0.17,
                text=name.replace("_", " ").title(),
                showarrow=False,
                font=dict(
                    size=10,
                    color="black"
                )
            )
            for i, name in enumerate(PIPELINE_COMPONENTS)
        ]
    )
    
    return fig

def run_next_component():
    """
    Runs the next component in the pipeline sequence.
    Updates the session state to reflect the execution.
    """
    if not st.session_state.executed_components:
        # If no components executed yet, start with the first one
        next_idx = 0
    else:
        # Find the index of the next component to execute
        next_idx = len(st.session_state.executed_components)
    
    # Make sure we don't exceed the number of components
    if next_idx >= len(PIPELINE_COMPONENTS):
        st.warning("All components have already been executed.")
        return
    
    # Get the next component to execute
    next_component = PIPELINE_COMPONENTS[next_idx]
    
    with st.spinner(f"Executing {next_component.replace('_', ' ').title()}..."):
        # Get input data (from session or generate)
        input_data = st.session_state.get('input_data', None)
        if input_data is None:
            from visualization.data_utils import load_or_generate_input_data
            input_data = load_or_generate_input_data(synthetic=True)
            st.session_state.input_data = input_data
        
        # Process data through pipeline up to this component
        if not st.session_state.pipeline_id:
            results = process_data_through_pipeline(input_data, next_component)
            st.session_state.pipeline_id = results.get('pipeline_id')
        else:
            # Process just this component if pipeline ID already exists
            results = process_data_through_pipeline(
                input_data, 
                next_component, 
                pipeline_id=st.session_state.pipeline_id
            )
        
        # Update session state
        st.session_state.executed_components.append(next_component)
        st.session_state.current_component = next_component
    
    st.success(f"{next_component.replace('_', ' ').title()} executed successfully!")
    st.rerun()

def on_component_click(trace, points, state):
    """
    Callback function for component clicks in the interactive diagram.
    
    Args:
        trace: The trace that was clicked
        points: The points that were clicked
        state: The state of the component
    """
    if not points.point_inds:
        return
    
    point_index = points.point_inds[0]
    component = trace.customdata[point_index]
    
    # Set the current component in session state
    st.session_state.current_component = component
    
    # Force a rerun to update the component details
    st.rerun()

if __name__ == "__main__":
    # For testing this module independently
    st.set_page_config(layout="wide", page_title="Interactive Pipeline Visualization")
    create_interactive_pipeline_view() 