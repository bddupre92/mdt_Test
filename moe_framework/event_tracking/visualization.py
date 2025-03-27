"""
Workflow Visualization for MoE Framework

This module provides visualization capabilities for workflow executions,
including static diagrams, interactive visualizations, and Mermaid diagrams.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from datetime import datetime

from .models import WorkflowExecution, WorkflowEvent, WorkflowComponentType

logger = logging.getLogger(__name__)

# Define color schemes for different component types
COMPONENT_COLORS = {
    WorkflowComponentType.DATA_LOADING: "#6495ED",  # Cornflower Blue
    WorkflowComponentType.QUALITY_ASSESSMENT: "#32CD32",  # Lime Green
    WorkflowComponentType.EXPERT_TRAINING: "#FF8C00",  # Dark Orange
    WorkflowComponentType.GATING_TRAINING: "#FF4500",  # Orange Red
    WorkflowComponentType.PREDICTION: "#9932CC",  # Dark Orchid
    WorkflowComponentType.INTEGRATION: "#FFD700",  # Gold
    WorkflowComponentType.WEIGHT_CALCULATION: "#FF69B4",  # Hot Pink
    WorkflowComponentType.EVALUATION: "#20B2AA",  # Light Sea Green
    WorkflowComponentType.CHECKPOINT: "#4682B4",  # Steel Blue
    WorkflowComponentType.SYSTEM: "#A9A9A9",  # Dark Gray
    # New component types
    WorkflowComponentType.DIFFERENTIAL_EVOLUTION: "#8A2BE2",  # Blue Violet
    WorkflowComponentType.EVOLUTION_STRATEGY: "#7B68EE",  # Medium Slate Blue
    WorkflowComponentType.ANT_COLONY_OPTIMIZATION: "#BA55D3",  # Medium Orchid
    WorkflowComponentType.GREY_WOLF_OPTIMIZER: "#9370DB",  # Medium Purple
    WorkflowComponentType.PHYSIOLOGICAL_EXPERT: "#DB7093",  # Pale Violet Red
    WorkflowComponentType.ENVIRONMENTAL_EXPERT: "#3CB371",  # Medium Sea Green
    WorkflowComponentType.BEHAVIORAL_EXPERT: "#F0E68C",  # Khaki
    WorkflowComponentType.MEDICATION_EXPERT: "#F4A460",  # Sandy Brown
    WorkflowComponentType.META_LEARNER: "#BDB76B",  # Dark Khaki
    WorkflowComponentType.GATING_NETWORK: "#CD5C5C",  # Indian Red
    WorkflowComponentType.OPTIMIZER_ANALYSIS: "#4169E1",  # Royal Blue
    WorkflowComponentType.OTHER: "#D3D3D3",  # Light Gray
}

class WorkflowVisualizer:
    """
    Generates visualizations of workflow executions in the MoE framework.
    """
    
    def __init__(self, tracking_dir: str):
        """
        Initialize the workflow visualizer.
        
        Args:
            tracking_dir: Directory containing workflow tracking data
        """
        self.tracking_dir = tracking_dir
        self.events_file = os.path.join(tracking_dir, 'events.json')
        
        # Create tracking directory if it doesn't exist
        os.makedirs(tracking_dir, exist_ok=True)
        
        logger.info(f"Initialized WorkflowVisualizer with output directory: {tracking_dir}")
        
    def get_events(self) -> List[Dict[str, Any]]:
        """
        Get all workflow events from the tracking directory.
        
        Returns:
            List of workflow events
        """
        events = []
        
        try:
            if os.path.exists(self.events_file):
                with open(self.events_file, 'r') as f:
                    events = json.load(f)
            else:
                logger.warning(f"No events file found at {self.events_file}")
                
        except Exception as e:
            logger.error(f"Error reading events file: {str(e)}")
            
        return events
        
    def add_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Add a new workflow event.
        
        Args:
            event_type: Type of the event
            event_data: Event data dictionary
            
        Returns:
            Success flag
        """
        try:
            # Load existing events
            events = self.get_events()
            
            # Add new event with timestamp
            event = {
                'type': event_type,
                'timestamp': datetime.now().isoformat(),
                'data': event_data
            }
            events.append(event)
            
            # Save updated events
            with open(self.events_file, 'w') as f:
                json.dump(events, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding event: {str(e)}")
            return False
            
    def get_timeline(self) -> List[Dict[str, Any]]:
        """
        Get timeline of workflow events.
        
        Returns:
            List of events sorted by timestamp
        """
        events = self.get_events()
        
        # Sort events by timestamp
        return sorted(events, key=lambda x: x.get('timestamp', ''))
        
    def get_stage_status(self) -> Dict[str, str]:
        """
        Get status of each workflow stage.
        
        Returns:
            Dictionary mapping stage names to their status
        """
        events = self.get_events()
        stages = {}
        
        for event in events:
            if event['type'].endswith('_STARTED'):
                stage = event['type'].replace('_STARTED', '')
                stages[stage] = 'In Progress'
            elif event['type'].endswith('_COMPLETED'):
                stage = event['type'].replace('_COMPLETED', '')
                stages[stage] = 'Completed'
            elif event['type'].endswith('_FAILED'):
                stage = event['type'].replace('_FAILED', '')
                stages[stage] = 'Failed'
                
        return stages
        
    def clear_events(self) -> bool:
        """
        Clear all workflow events.
        
        Returns:
            Success flag
        """
        try:
            if os.path.exists(self.events_file):
                os.remove(self.events_file)
            return True
        except Exception as e:
            logger.error(f"Error clearing events: {str(e)}")
            return False
        
    def create_execution_graph(self, workflow: WorkflowExecution) -> nx.DiGraph:
        """
        Create a directed graph for a workflow execution.
        
        Args:
            workflow: The workflow execution to visualize
            
        Returns:
            A NetworkX directed graph representing the workflow
        """
        G = nx.DiGraph()
        
        # Add nodes for each component
        for i, component in enumerate(workflow.components):
            node_id = f"comp_{i}"
            G.add_node(
                node_id,
                type="component",
                component=component.component.value,
                entry_time=component.entry_time,
                exit_time=component.exit_time,
                success=component.success,
                label=component.component.value
            )
        
        # Add nodes for each event and connect to components
        last_component = None
        for i, event in enumerate(workflow.events):
            node_id = f"event_{i}"
            G.add_node(
                node_id,
                type="event",
                event_type=event.event_type,
                component=event.component.value,
                timestamp=event.timestamp,
                success=event.success,
                label=event.event_type
            )
            
            # Find the closest component by time
            closest_component = None
            min_time_diff = float('inf')
            
            for j, component in enumerate(workflow.components):
                if component.component == event.component:
                    # Check if the event is within the component's time range
                    if (component.entry_time <= event.timestamp and 
                        (component.exit_time is None or event.timestamp <= component.exit_time)):
                        closest_component = f"comp_{j}"
                        break
            
            # Connect event to component
            if closest_component:
                G.add_edge(closest_component, node_id)
            elif last_component:
                # If no matching component, connect to the previous component
                G.add_edge(last_component, node_id)
            
            # Update last component
            if i > 0 and closest_component:
                last_component = closest_component
        
        # Add edges between components based on execution order
        sorted_components = sorted(
            [(i, comp) for i, comp in enumerate(workflow.components)],
            key=lambda x: x[1].entry_time
        )
        
        for i in range(len(sorted_components) - 1):
            current_idx, _ = sorted_components[i]
            next_idx, _ = sorted_components[i + 1]
            G.add_edge(f"comp_{current_idx}", f"comp_{next_idx}")
        
        return G
    
    def visualize_workflow(self, 
                          workflow: WorkflowExecution,
                          output_filename: Optional[str] = None,
                          figsize: Tuple[int, int] = (14, 10),
                          show_events: bool = True) -> str:
        """
        Create a visualization of a workflow execution.
        
        Args:
            workflow: The workflow execution to visualize
            output_filename: Optional filename for the output image
            figsize: Figure size (width, height) in inches
            show_events: Whether to show individual events
            
        Returns:
            Path to the output file
        """
        if not output_filename:
            output_filename = f"{workflow.workflow_id}.png"
            
        output_path = os.path.join(self.tracking_dir, output_filename)
        
        # Create the graph
        G = self.create_execution_graph(workflow)
        
        # Set up the figure
        plt.figure(figsize=figsize)
        
        # Define node positions using spring layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw nodes
        component_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'component']
        event_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'event']
        
        # Draw component nodes
        component_colors = [
            COMPONENT_COLORS[WorkflowComponentType(G.nodes[n]['component'])]
            for n in component_nodes
        ]
        
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=component_nodes,
            node_size=800,
            node_color=component_colors,
            alpha=0.8
        )
        
        # Draw event nodes if requested
        if show_events:
            event_colors = [
                COMPONENT_COLORS[WorkflowComponentType(G.nodes[n]['component'])]
                for n in event_nodes
            ]
            
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=event_nodes,
                node_size=300,
                node_color=event_colors,
                node_shape='o',
                alpha=0.6
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, 
            width=1.5, 
            arrowsize=20, 
            alpha=0.7,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Add labels
        component_labels = {
            n: G.nodes[n]['label'] for n in component_nodes
        }
        
        event_labels = {}
        if show_events:
            event_labels = {
                n: G.nodes[n]['event_type'] for n in event_nodes
            }
        
        # Draw component labels
        nx.draw_networkx_labels(
            G, pos, 
            labels=component_labels,
            font_size=10,
            font_weight='bold'
        )
        
        # Draw event labels if requested
        if show_events and event_labels:
            nx.draw_networkx_labels(
                G, pos, 
                labels=event_labels,
                font_size=8
            )
        
        # Create legend
        legend_patches = [
            mpatches.Patch(
                color=COMPONENT_COLORS[comp], 
                label=comp.value
            )
            for comp in WorkflowComponentType
            if comp in set(WorkflowComponentType(G.nodes[n]['component']) for n in G.nodes())
        ]
        
        plt.legend(
            handles=legend_patches,
            loc='upper right',
            title='Component Types'
        )
        
        # Set title and remove axes
        plt.title(f"Workflow Execution: {workflow.workflow_id}")
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved workflow visualization to: {output_path}")
        
        return output_path
    
    def generate_mermaid_workflow(self, 
                                workflow: WorkflowExecution,
                                output_filename: Optional[str] = None,
                                include_events: bool = False) -> str:
        """
        Generate a Mermaid.js diagram of a workflow execution.
        
        Args:
            workflow: The workflow execution to visualize
            output_filename: Optional filename for the output file
            include_events: Whether to include individual events
            
        Returns:
            Path to the output file
        """
        if not output_filename:
            output_filename = f"{workflow.workflow_id}.mmd"
            
        output_path = os.path.join(self.tracking_dir, output_filename)
        
        # Build the mermaid diagram
        mermaid_lines = ["graph TD"]
        
        # Add nodes for each component
        for i, component in enumerate(workflow.components):
            node_id = f"comp_{i}"
            shape = "([%s])" if component.success else "((%s))"
            label = shape % component.component.value
            
            # Add style based on component type
            style = f"{node_id}:::type{component.component.value.replace('_', '')}"
            
            mermaid_lines.append(f"    {node_id}{label}")
            mermaid_lines.append(f"    {style}")
        
        # Add nodes for events if requested
        if include_events:
            for i, event in enumerate(workflow.events):
                node_id = f"event_{i}"
                label = f"[{event.event_type}]"
                
                # Add style based on component type
                style = f"{node_id}:::type{event.component.value.replace('_', '')}"
                
                mermaid_lines.append(f"    {node_id}{label}")
                mermaid_lines.append(f"    {style}")
        
        # Add connections between components based on execution order
        sorted_components = sorted(
            [(i, comp) for i, comp in enumerate(workflow.components)],
            key=lambda x: x[1].entry_time
        )
        
        for i in range(len(sorted_components) - 1):
            current_idx, _ = sorted_components[i]
            next_idx, _ = sorted_components[i + 1]
            mermaid_lines.append(f"    comp_{current_idx} --> comp_{next_idx}")
        
        # Add connections for events if requested
        if include_events:
            for i, event in enumerate(workflow.events):
                # Find the closest component by time
                closest_component = None
                min_time_diff = float('inf')
                
                for j, component in enumerate(workflow.components):
                    if component.component == event.component:
                        # Check if the event is within the component's time range
                        if (component.entry_time <= event.timestamp and 
                            (component.exit_time is None or event.timestamp <= component.exit_time)):
                            closest_component = j
                            break
                
                if closest_component is not None:
                    mermaid_lines.append(f"    comp_{closest_component} --> event_{i}")
        
        # Add class definitions for styling
        mermaid_lines.append("    %% Class definitions for styling")
        for comp_type in WorkflowComponentType:
            type_name = comp_type.value.replace('_', '')
            color = COMPONENT_COLORS[comp_type].replace('#', '')
            mermaid_lines.append(f"    classDef type{type_name} fill:#{color},stroke:#333,stroke-width:1px")
        
        # Add title
        mermaid_lines.append(f"    %% Workflow: {workflow.workflow_id}")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(mermaid_lines))
        
        logger.info(f"Saved Mermaid workflow diagram to: {output_path}")
        
        return output_path
    
    def export_workflow_json(self, 
                           workflow: WorkflowExecution,
                           output_filename: Optional[str] = None) -> str:
        """
        Export a workflow execution to a JSON file for custom visualization.
        
        Args:
            workflow: The workflow execution to export
            output_filename: Optional filename for the output file
            
        Returns:
            Path to the output file
        """
        if not output_filename:
            output_filename = f"{workflow.workflow_id}_export.json"
            
        output_path = os.path.join(self.tracking_dir, output_filename)
        
        # Convert workflow to graph
        G = self.create_execution_graph(workflow)
        
        # Export to JSON format compatible with visualization libraries
        export_data = {
            "workflow_id": workflow.workflow_id,
            "start_time": workflow.start_time,
            "end_time": workflow.end_time,
            "success": workflow.success,
            "nodes": [],
            "links": []
        }
        
        # Add nodes
        for node_id, data in G.nodes(data=True):
            node_data = {
                "id": node_id,
                "type": data["type"],
                "label": data.get("label", ""),
                "success": data.get("success", True)
            }
            
            if data["type"] == "component":
                node_data["component"] = data["component"]
                node_data["color"] = COMPONENT_COLORS[WorkflowComponentType(data["component"])]
            else:
                node_data["event_type"] = data["event_type"]
                node_data["component"] = data["component"]
                node_data["color"] = COMPONENT_COLORS[WorkflowComponentType(data["component"])]
            
            export_data["nodes"].append(node_data)
        
        # Add links
        for source, target in G.edges():
            export_data["links"].append({
                "source": source,
                "target": target
            })
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported workflow execution to: {output_path}")
        
        return output_path
    
    def batch_visualize(self, 
                      workflows: List[WorkflowExecution],
                      output_prefix: Optional[str] = None,
                      show_events: bool = True) -> List[str]:
        """
        Create visualizations for multiple workflow executions.
        
        Args:
            workflows: List of workflow executions to visualize
            output_prefix: Optional prefix for output filenames
            show_events: Whether to show individual events
            
        Returns:
            List of paths to the output files
        """
        output_paths = []
        
        for i, workflow in enumerate(workflows):
            if output_prefix:
                output_filename = f"{output_prefix}_{i}.png"
            else:
                output_filename = None
                
            output_path = self.visualize_workflow(
                workflow=workflow,
                output_filename=output_filename,
                show_events=show_events
            )
            
            output_paths.append(output_path)
        
        return output_paths
    
    def create_timeline_visualization(self, 
                                   workflow: WorkflowExecution,
                                   output_filename: Optional[str] = None,
                                   figsize: Tuple[int, int] = (14, 8)) -> str:
        """
        Create a timeline visualization of the workflow execution.
        
        Args:
            workflow: The workflow execution to visualize
            output_filename: Optional filename for the output image
            figsize: Figure size (width, height) in inches
            
        Returns:
            Path to the output file
        """
        if not output_filename:
            output_filename = f"{workflow.workflow_id}_timeline.png"
            
        output_path = os.path.join(self.tracking_dir, output_filename)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort components by entry time
        sorted_components = sorted(
            workflow.components,
            key=lambda x: x.entry_time
        )
        
        # Extract timestamps and convert to numerical format
        timestamps = []
        for component in sorted_components:
            entry_time = datetime.fromisoformat(component.entry_time)
            timestamps.append(entry_time)
            
            if component.exit_time:
                exit_time = datetime.fromisoformat(component.exit_time)
                timestamps.append(exit_time)
        
        # Normalize timestamps
        if timestamps:
            min_time = min(timestamps)
            time_points = [(t - min_time).total_seconds() for t in timestamps]
            
            # Create y-positions for each component
            y_positions = {}
            for i, component in enumerate(sorted_components):
                y_positions[i] = i  # Use index as key instead of component object
            
            # Plot timeline bars
            for i, component in enumerate(sorted_components):
                entry_time = datetime.fromisoformat(component.entry_time)
                entry_seconds = (entry_time - min_time).total_seconds()
                
                if component.exit_time:
                    exit_time = datetime.fromisoformat(component.exit_time)
                    exit_seconds = (exit_time - min_time).total_seconds()
                    duration = exit_seconds - entry_seconds
                else:
                    # If no exit time, use a default duration
                    duration = 1.0
                
                color = COMPONENT_COLORS.get(component.component, COMPONENT_COLORS[WorkflowComponentType.OTHER])
                ax.barh(
                    i, 
                    duration, 
                    left=entry_seconds, 
                    height=0.5,
                    color=color, 
                    alpha=0.8,
                    label=component.component.value
                )
                
                # Add component label
                ax.text(
                    entry_seconds + duration / 2, 
                    i, 
                    component.component.value,
                    ha='center', 
                    va='center',
                    color='black', 
                    fontweight='bold',
                    fontsize=8
                )
            
            # Add events as points
            for event in workflow.events:
                event_time = datetime.fromisoformat(event.timestamp)
                event_seconds = (event_time - min_time).total_seconds()
                
                # Find the corresponding component (if any)
                for i, component in enumerate(sorted_components):
                    if component.component == event.component:
                        entry_time = datetime.fromisoformat(component.entry_time)
                        entry_seconds = (entry_time - min_time).total_seconds()
                        
                        if component.exit_time:
                            exit_time = datetime.fromisoformat(component.exit_time)
                            exit_seconds = (exit_time - min_time).total_seconds()
                        else:
                            exit_seconds = entry_seconds + 1.0
                            
                        if entry_seconds <= event_seconds <= exit_seconds:
                            y_pos = y_positions[i]  # Use index instead of component
                            
                            # Plot event point
                            ax.plot(
                                event_seconds, 
                                y_pos, 
                                'o', 
                                color='black', 
                                markersize=6, 
                                alpha=0.7
                            )
                            
                            # Add event label
                            ax.text(
                                event_seconds, 
                                y_pos + 0.3, 
                                event.event_type,
                                ha='center', 
                                va='bottom',
                                color='black', 
                                fontsize=6,
                                rotation=45
                            )
                            
                            break
            
            # Set y-axis labels and ticks
            ax.set_yticks(range(len(sorted_components)))
            ax.set_yticklabels([comp.component.value for comp in sorted_components])
            
            # Set x-axis label
            ax.set_xlabel('Time (seconds)')
            
            # Set title
            ax.set_title(f"Timeline: {workflow.workflow_id}")
            
            # Add grid
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Saved workflow timeline to: {output_path}")
            
            return output_path
        else:
            logger.warning("No components found for timeline visualization")
            return ""
    
    def visualize_optimizer_performance(self,
                                     workflow: WorkflowExecution,
                                     output_filename: Optional[str] = None,
                                     figsize: Tuple[int, int] = (16, 12)) -> Dict[str, str]:
        """
        Visualize the performance of evolutionary optimizers.
        
        Args:
            workflow: The workflow execution containing optimizer performance data
            output_filename: Optional base filename for output images
            figsize: Figure size (width, height) in inches
            
        Returns:
            Dictionary mapping visualization types to output file paths
        """
        if not workflow.optimizer_performances:
            logger.warning("No optimizer performance data found")
            return {}
            
        result_files = {}
        
        # Create visualizations for each optimizer
        for i, perf in enumerate(workflow.optimizer_performances):
            optimizer_type = perf.optimizer_type.value
            optimizer_id = perf.optimizer_id
            
            # Create convergence curve
            if perf.convergence_curve:
                if not output_filename:
                    conv_filename = f"{workflow.workflow_id}_{optimizer_id}_convergence.png"
                else:
                    conv_filename = f"{output_filename}_{optimizer_id}_convergence.png"
                    
                conv_path = os.path.join(self.tracking_dir, conv_filename)
                
                plt.figure(figsize=(10, 6))
                plt.plot(perf.convergence_curve, 'b-', linewidth=2)
                plt.title(f"Convergence Curve: {optimizer_type}")
                plt.xlabel("Iteration")
                plt.ylabel("Fitness Value")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(conv_path)
                plt.close()
                
                result_files["convergence"] = conv_path
                logger.info(f"Saved convergence curve to: {conv_path}")
            
            # Create parameter adaptation visualization
            if perf.parameters:
                if not output_filename:
                    param_filename = f"{workflow.workflow_id}_{optimizer_id}_parameters.png"
                else:
                    param_filename = f"{output_filename}_{optimizer_id}_parameters.png"
                    
                param_path = os.path.join(self.tracking_dir, param_filename)
                
                plt.figure(figsize=(12, 8))
                
                for j, (param_name, values) in enumerate(perf.parameters.items()):
                    if isinstance(values[0], (int, float)):
                        plt.plot(values, label=param_name)
                
                plt.title(f"Parameter Adaptation: {optimizer_type}")
                plt.xlabel("Iteration")
                plt.ylabel("Parameter Value")
                plt.legend(loc='best')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(param_path)
                plt.close()
                
                result_files["parameters"] = param_path
                logger.info(f"Saved parameter adaptation to: {param_path}")
            
            # Create exploration/exploitation visualization
            if perf.exploration_exploitation_ratio:
                if not output_filename:
                    explore_filename = f"{workflow.workflow_id}_{optimizer_id}_exploration.png"
                else:
                    explore_filename = f"{output_filename}_{optimizer_id}_exploration.png"
                    
                explore_path = os.path.join(self.tracking_dir, explore_filename)
                
                plt.figure(figsize=(10, 6))
                plt.plot(perf.exploration_exploitation_ratio, 'g-', linewidth=2)
                plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
                plt.title(f"Exploration/Exploitation Balance: {optimizer_type}")
                plt.xlabel("Iteration")
                plt.ylabel("Exploration Ratio")
                plt.ylim(0, 1)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(explore_path)
                plt.close()
                
                result_files["exploration"] = explore_path
                logger.info(f"Saved exploration/exploitation visualization to: {explore_path}")
            
            # Create diversity visualization
            if perf.diversity_history:
                if not output_filename:
                    div_filename = f"{workflow.workflow_id}_{optimizer_id}_diversity.png"
                else:
                    div_filename = f"{output_filename}_{optimizer_id}_diversity.png"
                    
                div_path = os.path.join(self.tracking_dir, div_filename)
                
                plt.figure(figsize=(10, 6))
                plt.plot(perf.diversity_history, 'r-', linewidth=2)
                plt.title(f"Population Diversity: {optimizer_type}")
                plt.xlabel("Iteration")
                plt.ylabel("Diversity")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(div_path)
                plt.close()
                
                result_files["diversity"] = div_path
                logger.info(f"Saved diversity visualization to: {div_path}")
            
            # Create summary dashboard
            if not output_filename:
                summary_filename = f"{workflow.workflow_id}_{optimizer_id}_summary.png"
            else:
                summary_filename = f"{output_filename}_{optimizer_id}_summary.png"
                
            summary_path = os.path.join(self.tracking_dir, summary_filename)
            
            fig = plt.figure(figsize=figsize)
            fig.suptitle(f"Optimizer Performance Summary: {optimizer_type}", fontsize=16)
            
            # Determine number of subplots needed
            num_plots = sum([
                bool(perf.convergence_curve),
                bool(perf.parameters),
                bool(perf.exploration_exploitation_ratio),
                bool(perf.diversity_history),
                True  # Always include a metadata text area
            ])
            
            rows = max(1, (num_plots + 1) // 2)
            cols = 2
            
            plot_idx = 1
            
            # Add convergence curve
            if perf.convergence_curve:
                ax = fig.add_subplot(rows, cols, plot_idx)
                ax.plot(perf.convergence_curve, 'b-', linewidth=2)
                ax.set_title("Convergence Curve")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Fitness Value")
                ax.grid(True, linestyle='--', alpha=0.7)
                plot_idx += 1
            
            # Add parameter adaptation
            if perf.parameters:
                ax = fig.add_subplot(rows, cols, plot_idx)
                
                for param_name, values in perf.parameters.items():
                    if isinstance(values[0], (int, float)):
                        ax.plot(values, label=param_name)
                
                ax.set_title("Parameter Adaptation")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Parameter Value")
                ax.legend(loc='best')
                ax.grid(True, linestyle='--', alpha=0.7)
                plot_idx += 1
            
            # Add exploration/exploitation
            if perf.exploration_exploitation_ratio:
                ax = fig.add_subplot(rows, cols, plot_idx)
                ax.plot(perf.exploration_exploitation_ratio, 'g-', linewidth=2)
                ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
                ax.set_title("Exploration/Exploitation Balance")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Exploration Ratio")
                ax.set_ylim(0, 1)
                ax.grid(True, linestyle='--', alpha=0.7)
                plot_idx += 1
            
            # Add diversity
            if perf.diversity_history:
                ax = fig.add_subplot(rows, cols, plot_idx)
                ax.plot(perf.diversity_history, 'r-', linewidth=2)
                ax.set_title("Population Diversity")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Diversity")
                ax.grid(True, linestyle='--', alpha=0.7)
                plot_idx += 1
            
            # Add metadata text
            ax = fig.add_subplot(rows, cols, plot_idx)
            ax.axis('off')
            metadata = (
                f"Optimizer: {optimizer_type}\n"
                f"ID: {optimizer_id}\n"
                f"Best Fitness: {perf.best_fitness:.6f}\n"
                f"Iterations: {perf.iterations}\n"
                f"Evaluations: {perf.evaluations}\n"
                f"Duration: {perf.duration:.3f} seconds"
            )
            ax.text(0.1, 0.5, metadata, fontsize=12, va='center')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(summary_path)
            plt.close()
            
            result_files["summary"] = summary_path
            logger.info(f"Saved optimizer summary to: {summary_path}")
        
        return result_files
    
    def visualize_expert_contributions(self,
                                     workflow: WorkflowExecution,
                                     output_filename: Optional[str] = None,
                                     figsize: Tuple[int, int] = (16, 12)) -> Dict[str, str]:
        """
        Visualize the contributions of expert models.
        
        Args:
            workflow: The workflow execution containing expert contribution data
            output_filename: Optional base filename for output images
            figsize: Figure size (width, height) in inches
            
        Returns:
            Dictionary mapping visualization types to output file paths
        """
        if not workflow.expert_contributions:
            logger.warning("No expert contribution data found")
            return {}
            
        result_files = {}
        
        # Group contributions by expert type
        expert_types = {}
        for contrib in workflow.expert_contributions:
            expert_type = contrib.expert_type.value
            if expert_type not in expert_types:
                expert_types[expert_type] = []
            expert_types[expert_type].append(contrib)
        
        # Create expert weight pie chart
        if not output_filename:
            weights_filename = f"{workflow.workflow_id}_expert_weights.png"
        else:
            weights_filename = f"{output_filename}_expert_weights.png"
            
        weights_path = os.path.join(self.tracking_dir, weights_filename)
        
        plt.figure(figsize=(10, 8))
        
        # Calculate average weights for each expert type
        avg_weights = {}
        for expert_type, contribs in expert_types.items():
            avg_weights[expert_type] = sum(c.weight for c in contribs) / len(contribs)
        
        # Create pie chart
        labels = list(avg_weights.keys())
        weights = list(avg_weights.values())
        colors = [COMPONENT_COLORS.get(WorkflowComponentType(label), COMPONENT_COLORS[WorkflowComponentType.OTHER]) for label in labels]
        
        plt.pie(weights, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title(f"Expert Model Contributions: {workflow.workflow_id}")
        plt.tight_layout()
        plt.savefig(weights_path)
        plt.close()
        
        result_files["weights"] = weights_path
        logger.info(f"Saved expert weights visualization to: {weights_path}")
        
        # Create confidence bar chart
        if not output_filename:
            conf_filename = f"{workflow.workflow_id}_expert_confidence.png"
        else:
            conf_filename = f"{output_filename}_expert_confidence.png"
            
        conf_path = os.path.join(self.tracking_dir, conf_filename)
        
        plt.figure(figsize=(12, 8))
        
        # Calculate average confidence for each expert type
        avg_conf = {}
        for expert_type, contribs in expert_types.items():
            avg_conf[expert_type] = sum(c.confidence for c in contribs) / len(contribs)
        
        # Create bar chart
        labels = list(avg_conf.keys())
        conf_values = list(avg_conf.values())
        colors = [COMPONENT_COLORS.get(WorkflowComponentType(label), COMPONENT_COLORS[WorkflowComponentType.OTHER]) for label in labels]
        
        plt.bar(labels, conf_values, color=colors)
        plt.ylim(0, 1)
        plt.xlabel('Expert Type')
        plt.ylabel('Confidence')
        plt.title(f"Expert Model Confidence: {workflow.workflow_id}")
        plt.tight_layout()
        plt.savefig(conf_path)
        plt.close()
        
        result_files["confidence"] = conf_path
        logger.info(f"Saved expert confidence visualization to: {conf_path}")
        
        # Create feature importance visualization
        if any(c.feature_importance for c in workflow.expert_contributions):
            if not output_filename:
                feat_filename = f"{workflow.workflow_id}_feature_importance.png"
            else:
                feat_filename = f"{output_filename}_feature_importance.png"
                
            feat_path = os.path.join(self.tracking_dir, feat_filename)
            
            # Collect feature importance data
            feature_importance = {}
            for expert_type, contribs in expert_types.items():
                # Combine feature importance from all contributions of this expert type
                combined_importance = {}
                for contrib in contribs:
                    for feature, importance in contrib.feature_importance.items():
                        if feature not in combined_importance:
                            combined_importance[feature] = []
                        combined_importance[feature].append(importance)
                
                # Calculate average importance for each feature
                avg_importance = {
                    feature: sum(values) / len(values)
                    for feature, values in combined_importance.items()
                }
                
                feature_importance[expert_type] = avg_importance
            
            # Create visualization
            fig, axes = plt.subplots(len(feature_importance), 1, figsize=figsize, sharex=True)
            if len(feature_importance) == 1:
                axes = [axes]
            
            for i, (expert_type, importances) in enumerate(feature_importance.items()):
                ax = axes[i]
                color = COMPONENT_COLORS.get(WorkflowComponentType(expert_type), COMPONENT_COLORS[WorkflowComponentType.OTHER])
                
                # Sort features by importance
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                features = [f[0] for f in sorted_features]
                values = [f[1] for f in sorted_features]
                
                ax.barh(features, values, color=color)
                ax.set_title(f"Feature Importance: {expert_type}")
                ax.set_xlabel('Importance')
                ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(feat_path)
            plt.close()
            
            result_files["feature_importance"] = feat_path
            logger.info(f"Saved feature importance visualization to: {feat_path}")
        
        return result_files
    
    def visualize_meta_learner_decisions(self,
                                       workflow: WorkflowExecution,
                                       output_filename: Optional[str] = None,
                                       figsize: Tuple[int, int] = (14, 10)) -> Dict[str, str]:
        """
        Visualize the decisions made by the meta-learner.
        
        Args:
            workflow: The workflow execution containing meta-learner decision data
            output_filename: Optional base filename for output images
            figsize: Figure size (width, height) in inches
            
        Returns:
            Dictionary mapping visualization types to output file paths
        """
        if not workflow.meta_learner_decisions:
            logger.warning("No meta-learner decision data found")
            return {}
            
        result_files = {}
        
        # Create algorithm selection frequency chart
        if not output_filename:
            freq_filename = f"{workflow.workflow_id}_algorithm_frequency.png"
        else:
            freq_filename = f"{output_filename}_algorithm_frequency.png"
            
        freq_path = os.path.join(self.tracking_dir, freq_filename)
        
        plt.figure(figsize=(12, 8))
        
        # Count algorithm selections
        algo_counts = {}
        for decision in workflow.meta_learner_decisions:
            algo = decision.selected_algorithm
            if algo not in algo_counts:
                algo_counts[algo] = 0
            algo_counts[algo] += 1
        
        # Create bar chart
        algorithms = list(algo_counts.keys())
        counts = list(algo_counts.values())
        
        plt.bar(algorithms, counts)
        plt.xlabel('Algorithm')
        plt.ylabel('Selection Count')
        plt.title(f"Algorithm Selection Frequency: {workflow.workflow_id}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(freq_path)
        plt.close()
        
        result_files["frequency"] = freq_path
        logger.info(f"Saved algorithm frequency visualization to: {freq_path}")
        
        # Create confidence distribution chart
        if not output_filename:
            conf_filename = f"{workflow.workflow_id}_selection_confidence.png"
        else:
            conf_filename = f"{output_filename}_selection_confidence.png"
            
        conf_path = os.path.join(self.tracking_dir, conf_filename)
        
        plt.figure(figsize=(10, 8))
        
        # Group confidence by algorithm
        algo_confidence = {}
        for decision in workflow.meta_learner_decisions:
            algo = decision.selected_algorithm
            if algo not in algo_confidence:
                algo_confidence[algo] = []
            algo_confidence[algo].append(decision.confidence)
        
        # Create box plot
        algorithms = list(algo_confidence.keys())
        confidence_data = [algo_confidence[algo] for algo in algorithms]
        
        plt.boxplot(confidence_data, labels=algorithms)
        plt.xlabel('Algorithm')
        plt.ylabel('Selection Confidence')
        plt.title(f"Algorithm Selection Confidence: {workflow.workflow_id}")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(conf_path)
        plt.close()
        
        result_files["confidence"] = conf_path
        logger.info(f"Saved selection confidence visualization to: {conf_path}")
        
        return result_files

def create_moe_flow_diagram(output_file: str = "moe_flow.png", figsize: Tuple[int, int] = (16, 12)):
    """
    Create a static diagram of the MoE framework flow.
    
    Args:
        output_file: Path to save the diagram
        figsize: Figure size (width, height) in inches
        
    Returns:
        Path to the output file
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Define the main components
    components = {
        "data_loader": "Data Loader",
        "data_splitter": "Data Splitter",
        "quality_assessment": "Quality Assessment",
        "expert_trainer": "Expert Trainer",
        "gating_trainer": "Gating Network Trainer",
        "prediction_engine": "Prediction Engine",
        "expert_predictor": "Expert Predictor",
        "gating_predictor": "Gating Network Predictor",
        "integration_layer": "Integration Layer",
        "evaluator": "Evaluator",
        "checkpoint_manager": "Checkpoint Manager"
    }
    
    # Add nodes for each component
    for comp_id, comp_name in components.items():
        G.add_node(comp_id, label=comp_name)
    
    # Define component types
    component_types = {
        "data_loader": WorkflowComponentType.DATA_LOADING,
        "data_splitter": WorkflowComponentType.DATA_LOADING,
        "quality_assessment": WorkflowComponentType.QUALITY_ASSESSMENT,
        "expert_trainer": WorkflowComponentType.EXPERT_TRAINING,
        "gating_trainer": WorkflowComponentType.GATING_TRAINING,
        "prediction_engine": WorkflowComponentType.PREDICTION,
        "expert_predictor": WorkflowComponentType.PREDICTION,
        "gating_predictor": WorkflowComponentType.PREDICTION,
        "integration_layer": WorkflowComponentType.INTEGRATION,
        "evaluator": WorkflowComponentType.EVALUATION,
        "checkpoint_manager": WorkflowComponentType.CHECKPOINT
    }
    
    # Add edges to define the workflow
    edges = [
        ("data_loader", "data_splitter"),
        ("data_splitter", "quality_assessment"),
        ("quality_assessment", "expert_trainer"),
        ("quality_assessment", "gating_trainer"),
        ("expert_trainer", "checkpoint_manager"),
        ("gating_trainer", "checkpoint_manager"),
        ("checkpoint_manager", "prediction_engine"),
        ("prediction_engine", "expert_predictor"),
        ("prediction_engine", "gating_predictor"),
        ("expert_predictor", "integration_layer"),
        ("gating_predictor", "integration_layer"),
        ("integration_layer", "evaluator"),
        ("evaluator", "checkpoint_manager")
    ]
    
    # Add the edges to the graph
    for source, target in edges:
        G.add_edge(source, target)
    
    # Set up the figure
    plt.figure(figsize=figsize)
    
    # Use a standard layout algorithm (hierarchical layout approximation)
    # Try multiple layouts for best results
    try:
        # Try the spring layout first with some tuning for better hierarchical structure
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    except Exception:
        # Fallback to shell layout
        pos = nx.shell_layout(G)
    
    # Define node colors based on component type
    node_colors = [COMPONENT_COLORS[component_types[node]] for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=1000,
        node_color=node_colors,
        alpha=0.8
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, 
        width=2.0, 
        arrowsize=20, 
        alpha=0.7,
        connectionstyle='arc3,rad=0.1'
    )
    
    # Draw labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels,
        font_size=12,
        font_weight='bold'
    )
    
    # Create legend
    unique_component_types = set(component_types.values())
    legend_patches = [
        mpatches.Patch(
            color=COMPONENT_COLORS[comp], 
            label=comp.value
        )
        for comp in unique_component_types
    ]
    
    plt.legend(
        handles=legend_patches,
        loc='upper right',
        title='Component Types'
    )
    
    # Set title and remove axes
    plt.title("MoE Framework Workflow")
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved MoE flow diagram to: {output_file}")
    
    return output_file 