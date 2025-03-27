"""
Workflow Dashboard for MoE Framework

This module provides Streamlit components for visualizing workflow executions
in an interactive dashboard.
"""

import os
import io
import json
import logging
import tempfile
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    # Try relative import first (when imported as part of a package)
    from .models import WorkflowExecution, WorkflowComponentType
    from .visualization import WorkflowVisualizer, COMPONENT_COLORS, create_moe_flow_diagram
except ImportError:
    # Fall back to absolute import (when run as a script)
    from moe_framework.event_tracking.models import WorkflowExecution, WorkflowComponentType
    from moe_framework.event_tracking.visualization import WorkflowVisualizer, COMPONENT_COLORS, create_moe_flow_diagram

logger = logging.getLogger(__name__)

def load_workflow_data(tracker_output_dir: str) -> List[WorkflowExecution]:
    """
    Load workflow data from a tracker output directory.
    
    Args:
        tracker_output_dir: Directory containing workflow JSON files
        
    Returns:
        List of workflow executions
    """
    workflows = []
    
    if not os.path.exists(tracker_output_dir):
        logger.warning(f"Tracker output directory does not exist: {tracker_output_dir}")
        return workflows
    
    for filename in os.listdir(tracker_output_dir):
        if filename.endswith(".json"):
            try:
                file_path = os.path.join(tracker_output_dir, filename)
                workflow = WorkflowExecution.load(file_path)
                workflows.append(workflow)
            except Exception as e:
                logger.error(f"Error loading workflow from {filename}: {str(e)}")
    
    # Sort workflows by start time
    workflows.sort(key=lambda w: w.start_time if w.start_time else "")
    
    return workflows

def render_workflow_dashboard(tracking_dir: str):
    """
    Render the MoE workflow dashboard.
    
    Args:
        tracking_dir: Directory containing workflow tracking data
    """
    # Skip page config as it's now handled in the main dashboard script
    
    # Display header
    st.title("MoE Framework Workflow Dashboard")
    
    # Load workflow data
    workflows = load_workflow_data(tracking_dir)
    
    if not workflows:
        st.warning(f"No workflow data found in {tracking_dir}. Run some MoE pipelines with tracking enabled to see visualizations.")
        st.info("Showing a static framework overview instead.")
        
        # Show static framework overview
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            flow_diagram_path = create_moe_flow_diagram(tmp.name)
            image = Image.open(flow_diagram_path)
            st.image(image, caption="MoE Framework Workflow", use_container_width=True)
        
        return
    
    # Sort workflows to prioritize those with optimizer/expert/meta-learner data
    def has_rich_data(workflow):
        has_optimizer = hasattr(workflow, 'optimizer_performances') and len(workflow.optimizer_performances) > 0
        has_expert = hasattr(workflow, 'expert_contributions') and len(workflow.expert_contributions) > 0
        has_meta = hasattr(workflow, 'meta_learner_decisions') and len(workflow.meta_learner_decisions) > 0
        return (has_optimizer or has_expert or has_meta, workflow.start_time)
    
    workflows.sort(key=has_rich_data, reverse=True)
    
    # Determine default workflow index - pick the latest workflow with rich data
    default_idx = 0
    if os.environ.get("SELECT_MOST_RECENT_WORKFLOW", "False").lower() == "true":
        for i, workflow in enumerate(workflows):
            if (hasattr(workflow, 'optimizer_performances') and workflow.optimizer_performances) or \
               (hasattr(workflow, 'expert_contributions') and workflow.expert_contributions) or \
               (hasattr(workflow, 'meta_learner_decisions') and workflow.meta_learner_decisions):
                default_idx = i
                break
    
    # Sidebar for workflow selection
    with st.sidebar:
        st.header("Select Workflow")
        
        selected_workflow_idx = st.selectbox(
            "Choose a workflow execution:",
            options=range(len(workflows)),
            format_func=lambda i: f"{workflows[i].workflow_id} ({workflows[i].start_time})" + 
                (" ✅ [has rich data]" if (hasattr(workflows[i], 'optimizer_performances') and workflows[i].optimizer_performances) or
                (hasattr(workflows[i], 'expert_contributions') and workflows[i].expert_contributions) or
                (hasattr(workflows[i], 'meta_learner_decisions') and workflows[i].meta_learner_decisions) else ""),
            index=default_idx
        )
        
        selected_workflow = workflows[selected_workflow_idx]
        
        # Display workflow details
        st.subheader("Workflow Details")
        st.write(f"**ID:** {selected_workflow.workflow_id}")
        st.write(f"**Start Time:** {selected_workflow.start_time}")
        st.write(f"**End Time:** {selected_workflow.end_time or 'Running'}")
        st.write(f"**Status:** {'✅ Success' if selected_workflow.success else '❌ Failed' if selected_workflow.end_time else '⏳ Running'}")
        st.write(f"**Components:** {len(selected_workflow.components)}")
        st.write(f"**Events:** {len(selected_workflow.events)}")
        
        # Visualization options
        st.header("Visualization Options")
        show_events = st.checkbox("Show Events", value=True)
        show_labels = st.checkbox("Show Labels", value=True)
        
        visualization_type = st.radio(
            "Visualization Type",
            ["Graph", "Timeline", "Mermaid Diagram", "Interactive", 
             "Optimizer Performance", "Expert Contributions", "Meta-Learner Decisions"]
        )
    
    # Main content area
    visualizer = WorkflowVisualizer(output_dir=tempfile.gettempdir())
    
    if visualization_type == "Graph":
        st.header("Workflow Graph Visualization")
        
        # Generate visualization
        with st.spinner("Generating workflow graph..."):
            output_path = visualizer.visualize_workflow(
                workflow=selected_workflow,
                show_events=show_events
            )
            
            image = Image.open(output_path)
            st.image(image, caption=f"Workflow: {selected_workflow.workflow_id}", use_container_width=True)
    
    elif visualization_type == "Timeline":
        st.header("Workflow Timeline Visualization")
        
        # Generate timeline
        with st.spinner("Generating workflow timeline..."):
            output_path = visualizer.create_timeline_visualization(
                workflow=selected_workflow
            )
            
            image = Image.open(output_path)
            st.image(image, caption=f"Timeline: {selected_workflow.workflow_id}", use_container_width=True)
    
    elif visualization_type == "Mermaid Diagram":
        st.header("Workflow Mermaid Diagram")
        
        # Generate Mermaid diagram
        with st.spinner("Generating Mermaid diagram..."):
            output_path = visualizer.generate_mermaid_workflow(
                workflow=selected_workflow,
                include_events=show_events
            )
            
            with open(output_path, 'r') as f:
                mermaid_code = f.read()
                
            st.markdown(f"```mermaid\n{mermaid_code}\n```")
    
    elif visualization_type == "Interactive":
        st.header("Interactive Workflow Visualization")
        
        # Generate interactive visualization using Streamlit's built-in capabilities
        with st.spinner("Generating interactive visualization..."):
            # Create the graph
            G = visualizer.create_execution_graph(selected_workflow)
            
            # Export to format compatible with Streamlit visualization
            export_data = visualizer.export_workflow_json(
                workflow=selected_workflow
            )
            
            with open(export_data, 'r') as f:
                graph_data = json.load(f)
            
            # Create nodes and edges DataFrames for Streamlit
            nodes_data = []
            for node in graph_data["nodes"]:
                nodes_data.append({
                    "id": node["id"],
                    "type": node["type"],
                    "label": node["label"],
                    "color": node["color"]
                })
            
            edges_data = []
            for link in graph_data["links"]:
                edges_data.append({
                    "source": link["source"],
                    "target": link["target"]
                })
            
            nodes_df = pd.DataFrame(nodes_data)
            edges_df = pd.DataFrame(edges_data)
            
            # Check if there's any data to visualize
            if nodes_df.empty:
                st.warning("No component or event data available for interactive visualization.")
            else:
                # Display graph using Streamlit
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if not show_events:
                        # Filter out event nodes if not showing events
                        nodes_df = nodes_df[nodes_df["type"] == "component"]
                        edges_df = edges_df[
                            edges_df["source"].isin(nodes_df["id"]) & 
                            edges_df["target"].isin(nodes_df["id"])
                        ]
                    
                    # Check again if we have nodes after filtering
                    if nodes_df.empty:
                        st.warning("No components to display after filtering events.")
                    else:
                        # Create a temporary position layout
                        pos = nx.spring_layout(G)
                        
                        # Add position to nodes
                        nodes_df["x"] = [pos[node_id][0] for node_id in nodes_df["id"]]
                        nodes_df["y"] = [pos[node_id][1] for node_id in nodes_df["id"]]
                        
                        # Plot interactive graph using matplotlib
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Draw edges
                        for _, edge in edges_df.iterrows():
                            source_node = nodes_df[nodes_df["id"] == edge["source"]].iloc[0]
                            target_node = nodes_df[nodes_df["id"] == edge["target"]].iloc[0]
                            
                            ax.arrow(
                                source_node["x"], source_node["y"],
                                target_node["x"] - source_node["x"],
                                target_node["y"] - source_node["y"],
                                head_width=0.03, head_length=0.05,
                                fc='k', ec='k', alpha=0.5
                            )
                        
                        # Draw nodes
                        for _, node in nodes_df.iterrows():
                            if node["type"] == "component":
                                size = 400
                            else:
                                size = 100
                                
                            ax.scatter(
                                node["x"], node["y"],
                                s=size, c=node["color"],
                                edgecolors='black', alpha=0.7
                            )
                            
                            if show_labels:
                                ax.text(
                                    node["x"], node["y"],
                                    node["label"],
                                    fontsize=8, ha='center', va='center'
                                )
                        
                        # Configure plot
                        ax.set_title(f"Interactive Workflow: {selected_workflow.workflow_id}")
                        ax.set_aspect('equal')
                        ax.axis('off')
                        
                        # Use custom event for selecting nodes
                        st.pyplot(fig)
                        
                        # Allow clicking on nodes (using coordinates)
                        if st.button("Select Node"):
                            st.info("Click on a node to view its details")
                
                with col2:
                    if not nodes_df.empty:
                        st.subheader("Node Details")
                        
                        # Display node list for selection
                        selected_node_id = st.selectbox(
                            "Select a node:",
                            options=nodes_df["id"].tolist(),
                            format_func=lambda node_id: f"{node_id} ({nodes_df[nodes_df['id'] == node_id]['label'].iloc[0]})"
                        )
                        
                        # Display node details
                        selected_node = nodes_df[nodes_df["id"] == selected_node_id].iloc[0]
                        node_data = G.nodes[selected_node_id]
                        
                        st.markdown(f"<div class='node-details'>", unsafe_allow_html=True)
                        
                        if node_data["type"] == "component":
                            st.markdown(f"<p class='component-label'>{node_data['label']}</p>", unsafe_allow_html=True)
                            st.write(f"**Type:** Component")
                            st.write(f"**Entry Time:** {node_data['entry_time']}")
                            st.write(f"**Exit Time:** {node_data['exit_time'] or 'Running'}")
                            st.write(f"**Status:** {'✅ Success' if node_data['success'] else '❌ Failed'}")
                        else:
                            st.markdown(f"<p class='event-label'>{node_data['event_type']}</p>", unsafe_allow_html=True)
                            st.write(f"**Type:** Event")
                            st.write(f"**Component:** {node_data['component']}")
                            st.write(f"**Timestamp:** {node_data['timestamp']}")
                            st.write(f"**Status:** {'✅ Success' if node_data['success'] else '❌ Failed'}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
    
    elif visualization_type == "Optimizer Performance":
        st.header("Optimizer Performance Analysis")
        
        # Check if there are any optimizer performances
        if not hasattr(selected_workflow, 'optimizer_performances') or not selected_workflow.optimizer_performances:
            st.warning("No optimizer performance data found for this workflow.")
        else:
            # Generate optimizer performance visualizations
            with st.spinner("Generating optimizer performance visualizations..."):
                result_files = visualizer.visualize_optimizer_performance(
                    workflow=selected_workflow
                )
                
                if result_files:
                    # Create tabs for different visualizations
                    tabs = st.tabs(["Summary", "Convergence", "Parameters", "Exploration", "Diversity"])
                    
                    with tabs[0]:  # Summary tab
                        if "summary" in result_files:
                            image = Image.open(result_files["summary"])
                            st.image(image, caption="Optimizer Performance Summary", use_container_width=True)
                    
                    with tabs[1]:  # Convergence tab
                        if "convergence" in result_files:
                            image = Image.open(result_files["convergence"])
                            st.image(image, caption="Convergence Curve", use_container_width=True)
                    
                    with tabs[2]:  # Parameters tab
                        if "parameters" in result_files:
                            image = Image.open(result_files["parameters"])
                            st.image(image, caption="Parameter Adaptation", use_container_width=True)
                    
                    with tabs[3]:  # Exploration tab
                        if "exploration" in result_files:
                            image = Image.open(result_files["exploration"])
                            st.image(image, caption="Exploration/Exploitation Balance", use_container_width=True)
                    
                    with tabs[4]:  # Diversity tab
                        if "diversity" in result_files:
                            image = Image.open(result_files["diversity"])
                            st.image(image, caption="Population Diversity", use_container_width=True)
                    
                    # Display optimizer details
                    with st.expander("Optimizer Details", expanded=False):
                        for i, perf in enumerate(selected_workflow.optimizer_performances):
                            st.subheader(f"Optimizer: {perf.optimizer_type.value}")
                            st.write(f"ID: {perf.optimizer_id}")
                            st.write(f"Best Fitness: {perf.best_fitness:.6f}")
                            st.write(f"Iterations: {perf.iterations}")
                            st.write(f"Evaluations: {perf.evaluations}")
                            st.write(f"Duration: {perf.duration:.3f} seconds")
                            st.divider()
                            
    elif visualization_type == "Expert Contributions":
        st.header("Expert Model Contributions")
        
        # Check if there are any expert contributions
        if not hasattr(selected_workflow, 'expert_contributions') or not selected_workflow.expert_contributions:
            st.warning("No expert contribution data found for this workflow.")
        else:
            # Generate expert contribution visualizations
            with st.spinner("Generating expert contribution visualizations..."):
                result_files = visualizer.visualize_expert_contributions(
                    workflow=selected_workflow
                )
                
                if result_files:
                    # Create tabs for different visualizations
                    tabs = st.tabs(["Weights", "Confidence", "Feature Importance"])
                    
                    with tabs[0]:  # Weights tab
                        if "weights" in result_files:
                            image = Image.open(result_files["weights"])
                            st.image(image, caption="Expert Model Weights", use_container_width=True)
                    
                    with tabs[1]:  # Confidence tab
                        if "confidence" in result_files:
                            image = Image.open(result_files["confidence"])
                            st.image(image, caption="Expert Model Confidence", use_container_width=True)
                    
                    with tabs[2]:  # Feature Importance tab
                        if "feature_importance" in result_files:
                            image = Image.open(result_files["feature_importance"])
                            st.image(image, caption="Expert Feature Importance", use_container_width=True)
                    
                    # Display expert details
                    with st.expander("Expert Details", expanded=False):
                        # Group contributions by expert type
                        expert_types = {}
                        for contrib in selected_workflow.expert_contributions:
                            expert_type = contrib.expert_type.value
                            if expert_type not in expert_types:
                                expert_types[expert_type] = []
                            expert_types[expert_type].append(contrib)
                        
                        for expert_type, contribs in expert_types.items():
                            st.subheader(f"Expert Type: {expert_type}")
                            
                            # Calculate average weights and confidence
                            avg_weight = sum(c.weight for c in contribs) / len(contribs)
                            avg_confidence = sum(c.confidence for c in contribs) / len(contribs)
                            
                            st.write(f"Average Weight: {avg_weight:.4f}")
                            st.write(f"Average Confidence: {avg_confidence:.4f}")
                            st.write(f"Number of Contributions: {len(contribs)}")
                            
                            # Show feature usage
                            if contribs[0].features_used:
                                st.write("Features Used:")
                                for feature in contribs[0].features_used:
                                    st.write(f"- {feature}")
                            
                            st.divider()
                            
    elif visualization_type == "Meta-Learner Decisions":
        st.header("Meta-Learner Decision Analysis")
        
        # Check if there are any meta-learner decisions
        if not hasattr(selected_workflow, 'meta_learner_decisions') or not selected_workflow.meta_learner_decisions:
            st.warning("No meta-learner decision data found for this workflow.")
        else:
            # Generate meta-learner decision visualizations
            with st.spinner("Generating meta-learner decision visualizations..."):
                result_files = visualizer.visualize_meta_learner_decisions(
                    workflow=selected_workflow
                )
                
                if result_files:
                    # Create tabs for different visualizations
                    tabs = st.tabs(["Algorithm Frequency", "Selection Confidence"])
                    
                    with tabs[0]:  # Algorithm Frequency tab
                        if "frequency" in result_files:
                            image = Image.open(result_files["frequency"])
                            st.image(image, caption="Algorithm Selection Frequency", use_container_width=True)
                    
                    with tabs[1]:  # Selection Confidence tab
                        if "confidence" in result_files:
                            image = Image.open(result_files["confidence"])
                            st.image(image, caption="Algorithm Selection Confidence", use_container_width=True)
                    
                    # Display meta-learner details
                    with st.expander("Meta-Learner Decision Details", expanded=False):
                        for i, decision in enumerate(selected_workflow.meta_learner_decisions):
                            st.subheader(f"Decision {i+1}")
                            st.write(f"ID: {decision.selection_id}")
                            st.write(f"Selected Algorithm: {decision.selected_algorithm}")
                            st.write(f"Confidence: {decision.confidence:.4f}")
                            
                            if decision.alternatives:
                                st.write("Alternative Algorithms:")
                                for algo, score in decision.alternatives.items():
                                    st.write(f"- {algo}: {score:.4f}")
                            
                            if decision.problem_features:
                                st.write("Problem Features:")
                                st.json(decision.problem_features)
                            
                            st.divider()
    
    # Additional tabs for different views
    tabs = st.tabs(["Component Analysis", "Event Timeline", "Raw Data"])
    
    # Component Analysis tab
    with tabs[0]:
        st.subheader("Component Execution Analysis")
        
        # Create a DataFrame of components
        component_data = []
        for comp in selected_workflow.components:
            # Calculate duration if available
            duration = None
            if comp.entry_time and comp.exit_time:
                from dateutil import parser
                entry_time = parser.parse(comp.entry_time)
                exit_time = parser.parse(comp.exit_time)
                duration = (exit_time - entry_time).total_seconds()
            
            component_data.append({
                "Component": comp.component.value,
                "Entry Time": comp.entry_time,
                "Exit Time": comp.exit_time or "Running",
                "Duration (s)": duration,
                "Status": "Success" if comp.success else "Failed" if comp.exit_time else "Running"
            })
        
        components_df = pd.DataFrame(component_data)
        
        # Check if components data is available
        if components_df.empty:
            st.warning("No component data available for this workflow.")
        else:
            # Display component statistics
            col1, col2 = st.columns(2)
            
            with col1:
                # Group by component type
                component_counts = components_df["Component"].value_counts()
                
                # Create a bar chart
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(
                    component_counts.index,
                    component_counts.values,
                    color=[COMPONENT_COLORS[WorkflowComponentType(c)] for c in component_counts.index]
                )
                
                # Add count labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.1,
                        f"{height:.0f}",
                        ha='center', va='bottom',
                        fontsize=8
                    )
                
                plt.xticks(rotation=45, ha='right')
                plt.title("Component Type Distribution")
                plt.tight_layout()
                
                st.pyplot(fig)
            
            with col2:
                # Create a pie chart of status distribution
                status_counts = components_df["Status"].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.pie(
                    status_counts.values,
                    labels=status_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#4CAF50', '#F44336', '#2196F3'] if 'Running' in status_counts.index else ['#4CAF50', '#F44336']
                )
                plt.title("Component Status Distribution")
                plt.tight_layout()
                
                st.pyplot(fig)
            
            # Display component duration if available
            if not components_df["Duration (s)"].isna().all():
                st.subheader("Component Duration Analysis")
                
                # Filter out rows with missing duration
                duration_df = components_df.dropna(subset=["Duration (s)"])
                
                # Group by component type and calculate average duration
                avg_duration = duration_df.groupby("Component")["Duration (s)"].mean().sort_values(ascending=False)
                
                # Create a bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(
                    avg_duration.index,
                    avg_duration.values,
                    color=[COMPONENT_COLORS[WorkflowComponentType(c)] for c in avg_duration.index]
                )
                
                # Add duration labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.1,
                        f"{height:.2f}s",
                        ha='center', va='bottom',
                        fontsize=8
                    )
                
                plt.xticks(rotation=45, ha='right')
                plt.title("Average Duration by Component Type")
                plt.tight_layout()
                
                st.pyplot(fig)
            
            # Display component table
            st.subheader("Component Details")
            st.dataframe(components_df)
    
    # Event Timeline tab
    with tabs[1]:
        st.subheader("Event Timeline")
        
        # Create a DataFrame of events
        event_data = []
        for event in selected_workflow.events:
            event_data.append({
                "Event Type": event.event_type,
                "Component": event.component.value,
                "Timestamp": event.timestamp,
                "Success": event.success
            })
        
        events_df = pd.DataFrame(event_data)
        
        # Check if events data is available
        if events_df.empty:
            st.warning("No event data available for this workflow.")
        else:
            # Convert timestamp to datetime
            from dateutil import parser
            events_df["Timestamp"] = events_df["Timestamp"].apply(lambda x: parser.parse(x))
            
            # Sort by timestamp
            events_df = events_df.sort_values("Timestamp")
            
            # Create a timeline visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Group events by component
            grouped_events = events_df.groupby("Component")
            
            # Assign y-position for each component
            component_positions = {}
            for i, (comp, _) in enumerate(grouped_events):
                component_positions[comp] = i
            
            # Plot events
            for comp, events in grouped_events:
                y_pos = component_positions[comp]
                x_pos = events["Timestamp"]
                color = COMPONENT_COLORS[WorkflowComponentType(comp)]
                
                ax.scatter(
                    x_pos, 
                    [y_pos] * len(events),
                    s=100,
                    c=color,
                    marker='o',
                    edgecolors='black',
                    alpha=0.7,
                    label=comp if comp not in ax.get_legend_handles_labels()[1] else ""
                )
            
            # Format the x-axis (time)
            ax.xaxis_date()
            fig.autofmt_xdate()
            
            # Format the y-axis (component types)
            ax.set_yticks(list(component_positions.values()))
            ax.set_yticklabels(list(component_positions.keys()))
            
            # Add labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Component Type')
            ax.set_title('Event Timeline')
            
            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper right')
            
            # Add grid lines
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display event table with filtering
            st.subheader("Event Details")
            
            # Add filters
            col1, col2 = st.columns(2)
            
            with col1:
                selected_components = st.multiselect(
                    "Filter by Component:",
                    options=sorted(events_df["Component"].unique()),
                    default=sorted(events_df["Component"].unique())
                )
            
            with col2:
                selected_events = st.multiselect(
                    "Filter by Event Type:",
                    options=sorted(events_df["Event Type"].unique()),
                    default=sorted(events_df["Event Type"].unique())
                )
            
            # Apply filters
            filtered_events = events_df[
                events_df["Component"].isin(selected_components) &
                events_df["Event Type"].isin(selected_events)
            ]
            
            # Display filtered events
            st.dataframe(filtered_events)
    
    # Raw Data tab
    with tabs[2]:
        st.subheader("Raw Workflow Data")
        
        # Display raw JSON data
        st.json(selected_workflow.to_dict())
        
        # Option to download workflow data
        workflow_json = json.dumps(selected_workflow.to_dict(), indent=2)
        st.download_button(
            label="Download Workflow JSON",
            data=workflow_json,
            file_name=f"{selected_workflow.workflow_id}.json",
            mime="application/json"
        )

def run_dashboard(tracker_output_dir: str = "./.workflow_tracking", port: int = 8507):
    """
    Run the workflow visualization dashboard as a standalone application.
    
    Args:
        tracker_output_dir: Directory containing workflow data
        port: Port to run the dashboard on
    """
    import sys
    
    # Prepare arguments for Streamlit
    sys.argv = [
        "streamlit",
        "run",
        __file__,
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.serverAddress", "localhost",
        "--theme.base", "light"
    ]
    
    # Set environment variables
    os.environ["TRACKER_OUTPUT_DIR"] = tracker_output_dir
    
    # Run Streamlit
    import streamlit.web.cli as stcli
    
    # Check if running as the main module
    if __name__ == "__main__":
        logger.info(f"Starting workflow dashboard on port {port}")
        logger.info(f"Using workflow data from {tracker_output_dir}")
        
        # Create dashboard function
        def _run_dashboard():
            render_workflow_dashboard(tracker_output_dir)
        
        # Run Streamlit with the dashboard function
        sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
        sys.exit(stcli._main_run_clExplicit(_run_dashboard))
    else:
        logger.info("To run the dashboard, call run_dashboard() directly")

if __name__ == "__main__":
    import re
    import sys
    
    # Get tracker output directory from environment variable
    tracker_output_dir = os.environ.get("TRACKER_OUTPUT_DIR", "./.workflow_tracking")
    
    # Render the dashboard
    render_workflow_dashboard(tracker_output_dir) 