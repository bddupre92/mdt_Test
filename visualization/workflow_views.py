"""
Workflow Visualization Components

This module provides visualization components for analyzing MoE framework workflow,
including pipeline stages, data flow, and training progress.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging

# Import data utilities
from visualization.data_utils import load_component_data

# Set up logging
logger = logging.getLogger(__name__)

def create_timeline(events: List[Dict[str, Any]]):
    """Create a timeline visualization of workflow events."""
    if not events:
        return
        
    # Convert events to DataFrame with proper timestamp handling
    df = pd.DataFrame(events)
    
    # Ensure we have timestamp information
    if 'timestamp' not in df.columns:
        # Try to extract timestamp from event data
        df['timestamp'] = df.apply(
            lambda row: row.get('data', {}).get('timestamp', None) 
            if isinstance(row.get('data'), dict) else None,
            axis=1
        )
    
    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort events by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate end times (each event ends when the next one starts)
    df['end_timestamp'] = df['timestamp'].shift(-1)
    # Set the last event's end time to be 1 minute after its start
    df.loc[df['end_timestamp'].isna(), 'end_timestamp'] = df['timestamp'] + pd.Timedelta(minutes=1)
    
    # Create timeline
    fig = px.timeline(
        df,
        x_start='timestamp',
        x_end='end_timestamp',
        y='type',
        title='Workflow Timeline',
        color='type'  # Color-code by event type
    )
    
    # Update layout for better visualization
    fig.update_layout(
        height=400,
        showlegend=True,
        xaxis_title='Time',
        yaxis_title='Event Type',
        # Improve readability
        yaxis={'categoryorder': 'category ascending'},
        legend_title_text='Event Types'
    )
    
    # Update traces for better visibility
    fig.update_traces(
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.8
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_status_indicators(pipeline_data: Dict[str, Any]):
    """Create status indicators for pipeline components."""
    st.subheader("Pipeline Status")
    
    # Create columns for status indicators
    cols = st.columns(4)
    
    # Pipeline initialization status
    with cols[0]:
        st.metric(
            "Pipeline Status",
            "Initialized" if pipeline_data['status'].get('initialized', False) else "Not Initialized"
        )
    
    # Data loading status
    with cols[1]:
        st.metric(
            "Data Status",
            "Loaded" if pipeline_data['status'].get('data_loaded', False) else "Not Loaded"
        )
    
    # Training status
    with cols[2]:
        st.metric(
            "Training Status",
            "Trained" if pipeline_data['status'].get('trained', False) else "Not Trained"
        )
    
    # Prediction status
    with cols[3]:
        st.metric(
            "Prediction Status",
            "Ready" if pipeline_data['status'].get('prediction_ready', False) else "Not Ready"
        )

def create_expert_status_table(training_data: Dict[str, Any]):
    """Create a table showing expert training status."""
    st.subheader("Expert Status")
    
    if not training_data.get('expert_status'):
        # Show initialized experts even if not trained
        if hasattr(st.session_state, 'moe_pipeline') and hasattr(st.session_state.moe_pipeline, 'experts'):
            expert_status = pd.DataFrame(
                {
                    'Status': ['Initialized' for _ in st.session_state.moe_pipeline.experts],
                    'Type': [type(expert).__name__ for expert in st.session_state.moe_pipeline.experts.values()]
                },
                index=[name for name in st.session_state.moe_pipeline.experts.keys()]
            )
            expert_status.index.name = 'Expert'
            
            # Display as table with custom formatting
            st.dataframe(
                expert_status.style.apply(
                    lambda x: ['background-color: #e6f3ff' if val == 'Initialized' else '' for val in x],
                    subset=['Status']
                )
            )
            
            # Show training instructions
            st.info("Experts are initialized but not yet trained. Use the training controls above to begin training.")
        else:
            st.info("No expert status information available")
        return
        
    # Create DataFrame for expert status
    expert_data = []
    for expert_id, status in training_data['expert_status'].items():
        expert_data.append({
            'Expert': expert_id,
            'Status': '✅ Trained' if status.get('trained', False) else '❌ Not Trained',
            'Type': status.get('type', 'Unknown')
        })
    
    expert_status = pd.DataFrame(expert_data)
    expert_status.set_index('Expert', inplace=True)
    
    # Display as table with custom formatting
    st.dataframe(
        expert_status.style.apply(
            lambda x: ['background-color: #e6ffe6' if val == '✅ Trained' 
                      else 'background-color: #ffe6e6' if val == '❌ Not Trained'
                      else '' for val in x],
            subset=['Status']
        )
    )
    
    # Show training status message
    if all(status.get('trained', False) for status in training_data['expert_status'].values()):
        st.success("All experts are trained and ready for predictions!")
    else:
        st.warning("Some experts are not yet trained. Use the training controls above to train them.")

def create_workflow_metrics(flow_data: Dict[str, Any]):
    """Create metrics showing workflow progress."""
    st.subheader("Workflow Progress")
    
    # Current stage
    st.metric("Current Stage", flow_data.get('current_stage', 'Not Started'))
    
    # Completed stages
    completed = flow_data.get('stages_completed', [])
    if completed:
        st.write("Completed Stages:")
        for stage in completed:
            st.success(stage)
    else:
        st.info("No stages completed yet")

def create_workflow_dashboard(
    pipeline_data: Dict[str, Any],
    flow_data: Dict[str, Any],
    training_data: Dict[str, Any]
):
    """Create the complete workflow dashboard."""
    st.title("MoE Framework Workflow Dashboard")
    
    # Create status indicators
    create_status_indicators(pipeline_data)
    
    # Show timeline of events
    st.header("Event Timeline")
    if pipeline_data.get('events'):
        create_timeline(pipeline_data['events'])
    else:
        st.info("No workflow events recorded yet")
    
    # Create two columns for expert status and workflow metrics
    col1, col2 = st.columns(2)
    
    with col1:
        create_expert_status_table(training_data)
        
    with col2:
        create_workflow_metrics(flow_data)
    
    # Show raw event data in expander
    with st.expander("View Raw Event Data"):
        if pipeline_data.get('events'):
            st.json(pipeline_data['events'])
        else:
            st.info("No events recorded")

def render_workflow_summary(pipeline_id: Optional[str] = None):
    """
    Renders a summary of the workflow execution including timing and performance metrics.
    
    Args:
        pipeline_id: Optional pipeline ID to load specific execution data
    """
    st.markdown("## Workflow Summary")
    
    # If no pipeline has been executed yet
    if pipeline_id is None:
        st.info("No pipeline has been executed yet. Run the pipeline to see workflow metrics.")
        return
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    # Generate metrics from pipeline components
    components = [
        "data_preprocessing",
        "feature_extraction",
        "missing_data_handling",
        "expert_training",
        "gating_network",
        "moe_integration",
        "output_generation"
    ]
    
    # Collect execution times for each component
    execution_times = {}
    total_time = 0
    success_count = 0
    
    for component in components:
        component_data = load_component_data(component, pipeline_id)
        if component_data and "execution_time" in component_data:
            execution_time = component_data["execution_time"]
            execution_times[component] = execution_time
            total_time += execution_time
            success_count += 1
    
    # Display summary metrics
    with col1:
        st.metric("Total Execution Time", f"{total_time:.2f}s")
    
    with col2:
        st.metric("Components Executed", f"{success_count}/{len(components)}")
    
    with col3:
        # Calculate completion percentage
        completion_pct = (success_count / len(components)) * 100
        st.metric("Pipeline Completion", f"{completion_pct:.0f}%")
    
    # Create execution timeline visualization
    if execution_times:
        st.markdown("### Execution Timeline")
        
        # Create a DataFrame for the timeline
        timeline_df = pd.DataFrame({
            "Component": [c.replace("_", " ").title() for c in execution_times.keys()],
            "Execution Time (s)": list(execution_times.values())
        })
        
        # Create horizontal bar chart
        fig = px.bar(
            timeline_df, 
            y="Component", 
            x="Execution Time (s)",
            orientation="h",
            color="Execution Time (s)",
            color_continuous_scale=px.colors.sequential.Blues,
            title="Component Execution Times"
        )
        
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    
    # Expert performance comparison if experts were trained
    expert_data = load_component_data("expert_training", pipeline_id)
    if expert_data and "experts" in expert_data:
        st.markdown("### Expert Model Performance")
        
        # Extract expert metrics
        experts = expert_data["experts"]
        
        if experts:
            # Create comparison chart
            metrics_to_compare = ["validation_score", "training_score", "feature_count"]
            expert_metrics = {}
            
            for expert_name, expert_info in experts.items():
                if not expert_info:
                    continue
                    
                expert_metrics[expert_name] = {
                    metric: expert_info.get(metric, 0) 
                    for metric in metrics_to_compare 
                    if metric in expert_info
                }
            
            if expert_metrics:
                # Convert to DataFrame for visualization
                expert_df = pd.DataFrame.from_dict(
                    {(i, j): expert_metrics[i][j] 
                    for i in expert_metrics.keys() 
                    for j in expert_metrics[i].keys()},
                    orient="index"
                ).reset_index()
                
                expert_df.columns = ["Expert", "Metric", "Value"]
                expert_df["Expert"] = expert_df["Expert"].apply(lambda x: x[0])
                expert_df["Metric"] = expert_df["Metric"].apply(lambda x: x[1])
                
                # Create grouped bar chart
                fig = px.bar(
                    expert_df, 
                    x="Expert", 
                    y="Value", 
                    color="Metric", 
                    barmode="group",
                    title="Expert Model Metrics Comparison"
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Expert Model",
                    yaxis_title="Value",
                    legend_title="Metric"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Show pipeline data flow visualization
    st.markdown("### Data Flow Visualization")
    render_data_flow_visualization(pipeline_id)

def render_data_flow_visualization(pipeline_id: Optional[str] = None):
    """
    Renders a visualization showing data flow through the pipeline stages.
    
    Args:
        pipeline_id: Optional pipeline ID to load specific execution data
    """
    # If no pipeline has been executed yet
    if pipeline_id is None:
        st.info("No pipeline has been executed yet. Run the pipeline to see data flow.")
        return
    
    # Define the pipeline components
    components = [
        "data_preprocessing",
        "feature_extraction",
        "missing_data_handling",
        "expert_training",
        "gating_network",
        "moe_integration",
        "output_generation"
    ]
    
    # Collect data sample sizes for each component
    data_sizes = {}
    data_quality = {}
    
    for component in components:
        component_data = load_component_data(component, pipeline_id)
        if component_data:
            # Get input and output data sizes if available
            if "input_shape" in component_data:
                data_sizes[f"{component}_input"] = component_data["input_shape"]
            if "output_shape" in component_data:
                data_sizes[f"{component}_output"] = component_data["output_shape"]
            
            # Get data quality metrics if available
            if "metrics" in component_data and "data_quality_score" in component_data["metrics"]:
                data_quality[component] = component_data["metrics"]["data_quality_score"]
    
    # Create Sankey diagram for data flow
    if data_sizes:
        # Prepare data for Sankey diagram
        labels = []
        source = []
        target = []
        value = []
        color = []
        
        # Add components as nodes
        for i, component in enumerate(components):
            # Add input node
            labels.append(f"{component.replace('_', ' ').title()} Input")
            
            # Add output node
            labels.append(f"{component.replace('_', ' ').title()} Output")
            
            # Link input to output
            source.append(i*2)
            target.append(i*2+1)
            input_size = data_sizes.get(f"{component}_input", [0, 0])
            output_size = data_sizes.get(f"{component}_output", [0, 0])
            
            # Use row count as value, or 100 as default
            value.append(input_size[0] if isinstance(input_size, (list, tuple)) and len(input_size) > 0 else 100)
            
            # Set color based on data quality if available
            quality = data_quality.get(component, 0.8)
            color.append(f"rgba(51, 102, 255, {quality})")
            
            # Link output to next component's input
            if i < len(components) - 1:
                source.append(i*2+1)
                target.append((i+1)*2)
                value.append(output_size[0] if isinstance(output_size, (list, tuple)) and len(output_size) > 0 else 100)
                color.append(f"rgba(51, 102, 255, {quality})")
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=color
            )
        )])
        
        fig.update_layout(
            title="Data Flow Through Pipeline Components",
            font=dict(size=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data flow information available for this pipeline execution.")

def render_execution_history(limit: int = 5):
    """
    Renders a history of pipeline executions with performance metrics.
    
    Args:
        limit: Maximum number of historical executions to display
    """
    st.markdown("## Execution History")
    
    # In a real implementation, we would load this from a database
    # For this demo, we'll generate some sample execution history
    if 'execution_history' not in st.session_state:
        # Generate sample execution history
        np.random.seed(42)
        
        history = []
        for i in range(limit):
            # Create a timestamp for each execution
            timestamp = datetime.now() - timedelta(days=i)
            
            # Generate random metrics
            metrics = {
                "accuracy": np.random.uniform(0.75, 0.95),
                "execution_time": np.random.uniform(10, 120),
                "components_executed": np.random.randint(5, 8),
                "data_size": np.random.randint(1000, 10000)
            }
            
            history.append({
                "pipeline_id": f"pipeline_{i+1}",
                "timestamp": timestamp,
                "metrics": metrics,
                "status": "completed"
            })
        
        st.session_state.execution_history = history
    
    # Display the execution history
    history = st.session_state.execution_history
    
    if not history:
        st.info("No previous executions found.")
        return
    
    # Create a DataFrame for the history
    history_df = pd.DataFrame([
        {
            "Pipeline ID": h["pipeline_id"],
            "Timestamp": h["timestamp"],
            "Status": h["status"],
            "Accuracy": h["metrics"]["accuracy"],
            "Execution Time (s)": h["metrics"]["execution_time"],
            "Components Executed": h["metrics"]["components_executed"],
            "Data Size": h["metrics"]["data_size"]
        }
        for h in history
    ])
    
    # Display the history table
    st.dataframe(history_df, use_container_width=True)
    
    # Create performance trend visualization
    st.markdown("### Performance Trends")
    
    # Create line chart for accuracy and execution time
    trends_df = history_df.copy()
    trends_df = trends_df.sort_values("Timestamp")
    
    fig = go.Figure()
    
    # Add accuracy line
    fig.add_trace(go.Scatter(
        x=trends_df["Timestamp"],
        y=trends_df["Accuracy"],
        mode="lines+markers",
        name="Accuracy",
        yaxis="y"
    ))
    
    # Add execution time line
    fig.add_trace(go.Scatter(
        x=trends_df["Timestamp"],
        y=trends_df["Execution Time (s)"],
        mode="lines+markers",
        name="Execution Time",
        yaxis="y2"
    ))
    
    # Set up layout with dual Y axes
    fig.update_layout(
        title="Pipeline Performance Trends",
        xaxis=dict(title="Execution Date"),
        yaxis=dict(
            title="Accuracy",
            range=[0, 1],
            side="left"
        ),
        yaxis2=dict(
            title="Execution Time (s)",
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_optimization_history():
    """
    Renders visualizations of model optimization history.
    Shows how hyperparameters and model configurations evolved over time.
    """
    st.markdown("## Optimization History")
    
    # In a real implementation, we would load this from a database
    # For this demo, we'll generate some sample optimization history
    if 'optimization_history' not in st.session_state:
        # Generate sample optimization history
        np.random.seed(42)
        
        history = []
        param_ranges = {
            "learning_rate": (0.001, 0.1),
            "max_depth": (3, 10),
            "num_estimators": (50, 200),
            "reg_alpha": (0.01, 1.0)
        }
        
        for i in range(20):
            # Generate random parameters
            params = {
                param: np.random.uniform(ranges[0], ranges[1]) if param != "max_depth" and param != "num_estimators"
                else int(np.random.uniform(ranges[0], ranges[1]))
                for param, ranges in param_ranges.items()
            }
            
            # Calculate a score based on parameters
            # This is a simplified model of how parameters affect performance
            score = (
                0.8 + 
                0.05 * (params["learning_rate"] - 0.001) / (0.1 - 0.001) +
                0.03 * (params["max_depth"] - 3) / (10 - 3) +
                0.07 * (params["num_estimators"] - 50) / (200 - 50) -
                0.04 * (params["reg_alpha"] - 0.01) / (1.0 - 0.01)
            )
            
            # Add some noise to the score
            score += np.random.normal(0, 0.02)
            
            # Clip score to reasonable range
            score = max(0.7, min(0.98, score))
            
            history.append({
                "iteration": i+1,
                "params": params,
                "score": score
            })
        
        st.session_state.optimization_history = history
    
    # Display the optimization history
    history = st.session_state.optimization_history
    
    if not history:
        st.info("No optimization history available.")
        return
    
    # Create a DataFrame for the history
    history_df = pd.DataFrame([
        {
            "Iteration": h["iteration"],
            "Score": h["score"],
            "Learning Rate": h["params"]["learning_rate"],
            "Max Depth": h["params"]["max_depth"],
            "Num Estimators": h["params"]["num_estimators"],
            "Reg Alpha": h["params"]["reg_alpha"]
        }
        for h in history
    ])
    
    # Create score progression visualization
    st.markdown("### Optimization Progress")
    
    # Create line chart for score progression
    fig = px.line(
        history_df,
        x="Iteration",
        y="Score",
        title="Model Score Progression During Optimization",
        markers=True
    )
    
    # Add best score marker
    best_idx = history_df["Score"].idxmax()
    best_iteration = history_df.loc[best_idx, "Iteration"]
    best_score = history_df.loc[best_idx, "Score"]
    
    fig.add_trace(go.Scatter(
        x=[best_iteration],
        y=[best_score],
        mode="markers",
        marker=dict(size=12, color="red", symbol="star"),
        name="Best Score"
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create parameter influence visualization
    st.markdown("### Parameter Influence Analysis")
    
    # Create scatter plots for each parameter
    parameters = ["Learning Rate", "Max Depth", "Num Estimators", "Reg Alpha"]
    
    # Let the user select which parameter to visualize
    selected_param = st.selectbox("Select parameter to analyze:", parameters)
    
    # Create scatter plot for selected parameter
    fig = px.scatter(
        history_df,
        x=selected_param,
        y="Score",
        color="Score",
        size="Score",
        color_continuous_scale=px.colors.sequential.Viridis,
        title=f"Impact of {selected_param} on Model Performance"
    )
    
    # Add trendline
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    # Add best parameter marker
    best_param = history_df.loc[best_idx, selected_param]
    
    fig.add_trace(go.Scatter(
        x=[best_param],
        y=[best_score],
        mode="markers",
        marker=dict(size=15, color="red", symbol="star-diamond"),
        name="Best Configuration"
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show parallel coordinates plot for parameter relationships
    st.markdown("### Parameter Relationships")
    
    fig = px.parallel_coordinates(
        history_df,
        dimensions=["Learning Rate", "Max Depth", "Num Estimators", "Reg Alpha", "Score"],
        color="Score",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    st.plotly_chart(fig, use_container_width=True) 