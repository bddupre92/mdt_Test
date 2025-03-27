# MoE Framework Workflow Event Tracking

This module provides a comprehensive workflow tracking system for the MoE (Mixture of Experts) framework. It allows tracking, visualizing, and analyzing workflow execution to gain insights into the operation of the framework.

## Overview

The workflow event tracking system integrates with the existing event system to track workflow execution, providing detailed insights into component interactions, execution flow, and performance characteristics.

Key features:
- Event-based tracking of workflow execution
- Component-level timing and success metrics
- Multiple visualization formats (graph, timeline, JSON)
- Interactive Streamlit dashboard for analysis
- Mermaid diagram generation for documentation

## Architecture

The system consists of the following components:

1. **WorkflowTracker**: Hooks into the event system to track workflow execution
2. **WorkflowVisualizer**: Generates visualizations of workflow executions
3. **Dashboard**: Interactive Streamlit interface for exploring workflow data
4. **Data Models**: Structured representations of workflow entities

## Usage

### Basic Usage

```python
from moe_framework.event_tracking import WorkflowTracker

# Initialize tracker
tracker = WorkflowTracker(output_dir="./.workflow_tracking")

# Track a pipeline
pipeline = MyMoEPipeline()
tracked_pipeline = tracker.track_moe_pipeline(pipeline)

# Set event manager
tracker.event_manager = pipeline.event_manager

# Start workflow tracking
workflow = tracker.start_workflow("my_workflow")

# Run the workflow
tracked_pipeline.load_data("data.csv")
tracked_pipeline.train()
tracked_pipeline.predict([1, 2, 3])

# Complete the workflow (usually done automatically by checkpoint events)
tracker.complete_workflow(success=True)
```

### Visualization

```python
from moe_framework.event_tracking import WorkflowVisualizer

# Create visualizer
visualizer = WorkflowVisualizer(output_dir="./visualizations")

# Generate visualizations
graph_path = visualizer.visualize_workflow(workflow)
timeline_path = visualizer.create_timeline_visualization(workflow)
mermaid_path = visualizer.generate_mermaid_workflow(workflow)
json_path = visualizer.export_workflow_json(workflow)
```

### Dashboard

```python
from moe_framework.event_tracking import render_workflow_dashboard

# Run dashboard
render_workflow_dashboard(tracker_output_dir="./.workflow_tracking")

# Or use the standalone function
from moe_framework.event_tracking.dashboard import run_dashboard
run_dashboard(tracker_output_dir="./.workflow_tracking", port=8507)
```

### Component Tracking with Decorators

```python
from moe_framework.event_tracking import WorkflowTracker
from moe_framework.event_tracking.models import WorkflowComponentType

tracker = WorkflowTracker()

@tracker.track_component(WorkflowComponentType.DATA_LOADING)
def load_data(path):
    # Load data...
    return data

# Component will be tracked automatically
data = load_data("data.csv")
```

## Dashboard Features

The interactive dashboard provides:

1. **Graph Visualization**: Network diagram of component relationships
2. **Timeline View**: Temporal view of component execution
3. **Component Analysis**: Statistics and charts about component execution
4. **Event Timeline**: Visualization of all events in the workflow
5. **Raw Data View**: Access to underlying JSON data

## Integration with MoE Framework

The event tracking system integrates seamlessly with the MoE framework by:
- Hooking into the existing event system
- Tracking execution of pipelines, experts, and gating networks
- Providing insights into performance and behavior

## Dependencies

- NetworkX: For graph creation and analysis
- Matplotlib: For static visualizations
- Streamlit: For interactive dashboard
- Python 3.7+: Required for dataclasses and typing support

## Examples

See the `example.py` file for a complete demonstration of the workflow tracking system. 