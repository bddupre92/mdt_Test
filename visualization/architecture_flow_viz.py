#!/usr/bin/env python
"""
MoE Architecture Flow Visualization

This script creates visualizations that accurately represent the architecture and data flow
of the Mixture of Experts (MoE) framework, showing how data moves through the system
from input to prediction.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
import tempfile
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from matplotlib.path import Path as MatplotlibPath
import matplotlib.patches as patches

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Expert domain colors
EXPERT_COLORS = {
    "physiological": "#2C7BB6",  # Blue
    "environmental": "#7FBC41",  # Green
    "behavioral": "#D7301F",     # Red
    "medication_history": "#B15928"  # Brown
}

# Component colors
COMPONENT_COLORS = {
    "data_connector": "#8073AC",  # Purple
    "data_quality": "#FDB863",    # Orange
    "expert": "#B2DF8A",          # Light green
    "gating": "#A6CEE3",          # Light blue
    "integration": "#FB9A99",     # Light red
    "meta_learner": "#FDBF6F",    # Light orange
    "output": "#CAB2D6"           # Light purple
}

def load_checkpoint(checkpoint_path):
    """Load checkpoint data from JSON file."""
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded checkpoint data from {checkpoint_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading checkpoint data: {str(e)}")
        return None

def visualize_architecture_flow(checkpoint_data, output_dir):
    """Create a visualization of the MoE architecture and data flow."""
    if checkpoint_data:
        # Extract architecture information if available
        architecture_info = checkpoint_data.get("architecture", {})
    else:
        architecture_info = {}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define component positions
    positions = {
        "input": (1, 8),
        "data_connector": (3, 8),
        "data_quality": (3, 6),
        "expert_physiological": (5, 9),
        "expert_environmental": (5, 7.5),
        "expert_behavioral": (5, 6),
        "expert_medication": (5, 4.5),
        "gating": (3, 3),
        "integration": (7, 6),
        "meta_learner": (5, 2),
        "output": (9, 6)
    }
    
    # Define component sizes
    sizes = {
        "input": (1.2, 0.8),
        "data_connector": (1.5, 0.8),
        "data_quality": (1.5, 0.8),
        "expert_physiological": (1.5, 0.8),
        "expert_environmental": (1.5, 0.8),
        "expert_behavioral": (1.5, 0.8),
        "expert_medication": (1.5, 0.8),
        "gating": (1.5, 1.0),
        "integration": (1.5, 1.0),
        "meta_learner": (1.5, 0.8),
        "output": (1.2, 0.8)
    }
    
    # Draw components
    components = {}
    
    # Input data
    input_rect = patches.Rectangle(
        (positions["input"][0] - sizes["input"][0]/2, positions["input"][1] - sizes["input"][1]/2),
        sizes["input"][0], sizes["input"][1], 
        linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.8
    )
    ax.add_patch(input_rect)
    ax.text(positions["input"][0], positions["input"][1], "Input Data", 
            ha='center', va='center', fontweight='bold')
    components["input"] = input_rect
    
    # Data connector
    data_connector_rect = patches.Rectangle(
        (positions["data_connector"][0] - sizes["data_connector"][0]/2, 
         positions["data_connector"][1] - sizes["data_connector"][1]/2),
        sizes["data_connector"][0], sizes["data_connector"][1], 
        linewidth=1, edgecolor='black', facecolor=COMPONENT_COLORS["data_connector"], alpha=0.8
    )
    ax.add_patch(data_connector_rect)
    ax.text(positions["data_connector"][0], positions["data_connector"][1], "Data Connector", 
            ha='center', va='center', fontweight='bold')
    components["data_connector"] = data_connector_rect
    
    # Data quality
    data_quality_rect = patches.Rectangle(
        (positions["data_quality"][0] - sizes["data_quality"][0]/2, 
         positions["data_quality"][1] - sizes["data_quality"][1]/2),
        sizes["data_quality"][0], sizes["data_quality"][1], 
        linewidth=1, edgecolor='black', facecolor=COMPONENT_COLORS["data_quality"], alpha=0.8
    )
    ax.add_patch(data_quality_rect)
    ax.text(positions["data_quality"][0], positions["data_quality"][1], "Data Quality\nAssessment", 
            ha='center', va='center', fontweight='bold')
    components["data_quality"] = data_quality_rect
    
    # Expert models
    expert_components = []
    for expert_type, y_pos in [
        ("physiological", positions["expert_physiological"][1]),
        ("environmental", positions["expert_environmental"][1]),
        ("behavioral", positions["expert_behavioral"][1]),
        ("medication_history", positions["expert_medication"][1])
    ]:
        component_key = f"expert_{expert_type}"
        expert_rect = patches.Rectangle(
            (positions[component_key][0] - sizes[component_key][0]/2, 
             positions[component_key][1] - sizes[component_key][1]/2),
            sizes[component_key][0], sizes[component_key][1], 
            linewidth=1, edgecolor='black', facecolor=EXPERT_COLORS[expert_type], alpha=0.8
        )
        ax.add_patch(expert_rect)
        ax.text(positions[component_key][0], positions[component_key][1], 
                f"{expert_type.replace('_', ' ').title()}\nExpert", 
                ha='center', va='center', fontweight='bold')
        components[component_key] = expert_rect
        expert_components.append(expert_rect)
    
    # Gating network
    gating_rect = patches.Rectangle(
        (positions["gating"][0] - sizes["gating"][0]/2, 
         positions["gating"][1] - sizes["gating"][1]/2),
        sizes["gating"][0], sizes["gating"][1], 
        linewidth=1, edgecolor='black', facecolor=COMPONENT_COLORS["gating"], alpha=0.8
    )
    ax.add_patch(gating_rect)
    ax.text(positions["gating"][0], positions["gating"][1], "Gating Network", 
            ha='center', va='center', fontweight='bold')
    components["gating"] = gating_rect
    
    # Integration layer
    integration_rect = patches.Rectangle(
        (positions["integration"][0] - sizes["integration"][0]/2, 
         positions["integration"][1] - sizes["integration"][1]/2),
        sizes["integration"][0], sizes["integration"][1], 
        linewidth=1, edgecolor='black', facecolor=COMPONENT_COLORS["integration"], alpha=0.8
    )
    ax.add_patch(integration_rect)
    ax.text(positions["integration"][0], positions["integration"][1], "Integration\nLayer", 
            ha='center', va='center', fontweight='bold')
    components["integration"] = integration_rect
    
    # Meta-learner
    meta_learner_rect = patches.Rectangle(
        (positions["meta_learner"][0] - sizes["meta_learner"][0]/2, 
         positions["meta_learner"][1] - sizes["meta_learner"][1]/2),
        sizes["meta_learner"][0], sizes["meta_learner"][1], 
        linewidth=1, edgecolor='black', facecolor=COMPONENT_COLORS["meta_learner"], alpha=0.8
    )
    ax.add_patch(meta_learner_rect)
    ax.text(positions["meta_learner"][0], positions["meta_learner"][1], "Meta-Learner", 
            ha='center', va='center', fontweight='bold')
    components["meta_learner"] = meta_learner_rect
    
    # Output
    output_rect = patches.Rectangle(
        (positions["output"][0] - sizes["output"][0]/2, 
         positions["output"][1] - sizes["output"][1]/2),
        sizes["output"][0], sizes["output"][1], 
        linewidth=1, edgecolor='black', facecolor=COMPONENT_COLORS["output"], alpha=0.8
    )
    ax.add_patch(output_rect)
    ax.text(positions["output"][0], positions["output"][1], "Prediction", 
            ha='center', va='center', fontweight='bold')
    components["output"] = output_rect
    
    # Draw connections
    connections = [
        # Data flow
        ("input", "data_connector"),
        ("data_connector", "data_quality"),
        ("data_connector", "expert_physiological"),
        ("data_connector", "expert_environmental"),
        ("data_connector", "expert_behavioral"),
        ("data_connector", "expert_medication"),
        ("data_quality", "gating"),
        ("expert_physiological", "integration"),
        ("expert_environmental", "integration"),
        ("expert_behavioral", "integration"),
        ("expert_medication", "integration"),
        ("gating", "integration"),
        ("integration", "output"),
        
        # Meta-learner connections
        ("data_quality", "meta_learner"),
        ("meta_learner", "gating")
    ]
    
    for start, end in connections:
        start_pos = positions[start]
        end_pos = positions[end]
        
        # Adjust connection points based on component positions
        if start == "data_connector" and end.startswith("expert_"):
            # Connect from right side of data connector to left side of expert
            start_x = start_pos[0] + sizes[start][0]/2
            end_x = end_pos[0] - sizes[end][0]/2
        elif start.startswith("expert_") and end == "integration":
            # Connect from right side of expert to left side of integration
            start_x = start_pos[0] + sizes[start][0]/2
            end_x = end_pos[0] - sizes[end][0]/2
        elif start == "gating" and end == "integration":
            # Connect from top of gating to bottom of integration
            arrow = patches.FancyArrowPatch(
                (start_pos[0], start_pos[1] + sizes[start][1]/2),
                (end_pos[0], end_pos[1] - sizes[end][1]/2),
                connectionstyle="arc3,rad=0.3",
                arrowstyle="->",
                mutation_scale=15,
                linewidth=1.5,
                color="black"
            )
            ax.add_patch(arrow)
            continue
        elif start == "meta_learner" and end == "gating":
            # Connect from right of meta-learner to left of gating
            arrow = patches.FancyArrowPatch(
                (start_pos[0] + sizes[start][0]/2, start_pos[1]),
                (end_pos[0] - sizes[end][0]/2, end_pos[1]),
                connectionstyle="arc3,rad=0.2",
                arrowstyle="->",
                mutation_scale=15,
                linewidth=1.5,
                color="black"
            )
            ax.add_patch(arrow)
            continue
        else:
            # Default connection points
            start_x, end_x = start_pos[0], end_pos[0]
        
        # Draw arrow
        arrow = patches.FancyArrowPatch(
            (start_x, start_pos[1]),
            (end_x, end_pos[1]),
            arrowstyle="->",
            mutation_scale=15,
            linewidth=1.5,
            color="black"
        )
        ax.add_patch(arrow)
    
    # Add title
    plt.suptitle("MoE Framework Architecture and Data Flow", fontsize=16, y=0.98)
    
    # Add description
    description = (
        "This diagram shows the architecture and data flow of the Mixture of Experts (MoE) framework.\n"
        "Data flows from input through data connectors to expert models and the gating network.\n"
        "The gating network determines weights for each expert, and the integration layer combines\n"
        "expert predictions to produce the final output. The Meta-Learner optimizes the gating network based on data quality."
    )
    plt.figtext(0.5, 0.02, description, ha='center', fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # Save visualization
    output_path = os.path.join(output_dir, "architecture_flow.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Created architecture flow visualization: {output_path}")
    return output_path

def visualize_data_flow_sequence(checkpoint_data, output_dir):
    """Create a visualization of the data flow sequence in the MoE framework."""
    # Create figure
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [1, 1, 1.5, 1, 1]})
    
    # Step 1: Data Ingestion
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Draw components
    input_rect = patches.Rectangle((1, 1), 2, 1, linewidth=1, 
                                  edgecolor='black', facecolor='lightgray', alpha=0.8)
    ax.add_patch(input_rect)
    ax.text(2, 1.5, "Input Data", ha='center', va='center', fontweight='bold')
    
    connector_rect = patches.Rectangle((5, 1), 2, 1, linewidth=1, 
                                      edgecolor='black', facecolor=COMPONENT_COLORS["data_connector"], alpha=0.8)
    ax.add_patch(connector_rect)
    ax.text(6, 1.5, "Data Connector", ha='center', va='center', fontweight='bold')
    
    # Draw arrow
    arrow = patches.FancyArrowPatch(
        (3, 1.5), (5, 1.5),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.5,
        color="black"
    )
    ax.add_patch(arrow)
    
    # Add step title
    ax.text(0.5, 2.5, "Step 1: Data Ingestion", fontsize=12, fontweight='bold')
    
    # Step 2: Data Quality Assessment
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Draw components
    connector_rect = patches.Rectangle((1, 1), 2, 1, linewidth=1, 
                                      edgecolor='black', facecolor=COMPONENT_COLORS["data_connector"], alpha=0.8)
    ax.add_patch(connector_rect)
    ax.text(2, 1.5, "Data Connector", ha='center', va='center', fontweight='bold')
    
    quality_rect = patches.Rectangle((5, 1), 2, 1, linewidth=1, 
                                    edgecolor='black', facecolor=COMPONENT_COLORS["data_quality"], alpha=0.8)
    ax.add_patch(quality_rect)
    ax.text(6, 1.5, "Data Quality\nAssessment", ha='center', va='center', fontweight='bold')
    
    # Draw arrow
    arrow = patches.FancyArrowPatch(
        (3, 1.5), (5, 1.5),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.5,
        color="black"
    )
    ax.add_patch(arrow)
    
    # Add step title
    ax.text(0.5, 2.5, "Step 2: Data Quality Assessment", fontsize=12, fontweight='bold')
    
    # Step 3: Expert Model Predictions
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Draw data connector
    connector_rect = patches.Rectangle((1, 2), 2, 1, linewidth=1, 
                                      edgecolor='black', facecolor=COMPONENT_COLORS["data_connector"], alpha=0.8)
    ax.add_patch(connector_rect)
    ax.text(2, 2.5, "Data Connector", ha='center', va='center', fontweight='bold')
    
    # Draw expert models
    expert_types = ["physiological", "environmental", "behavioral", "medication_history"]
    expert_labels = ["Physiological\nExpert", "Environmental\nExpert", 
                     "Behavioral\nExpert", "Medication\nExpert"]
    
    for i, (expert_type, label) in enumerate(zip(expert_types, expert_labels)):
        y_pos = 0.5 + i
        expert_rect = patches.Rectangle((5, y_pos), 2, 0.8, linewidth=1, 
                                       edgecolor='black', facecolor=EXPERT_COLORS[expert_type], alpha=0.8)
        ax.add_patch(expert_rect)
        ax.text(6, y_pos + 0.4, label, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Draw arrow from connector to expert
        arrow = patches.FancyArrowPatch(
            (3, 2.5), (5, y_pos + 0.4),
            arrowstyle="->",
            connectionstyle=f"arc3,rad={0.1 * (i - 1.5)}",
            mutation_scale=15,
            linewidth=1.5,
            color="black"
        )
        ax.add_patch(arrow)
    
    # Add step title
    ax.text(0.5, 4.5, "Step 3: Expert Model Predictions", fontsize=12, fontweight='bold')
    
    # Step 4: Gating Network Weighting
    ax = axes[3]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Draw components
    quality_rect = patches.Rectangle((1, 1), 2, 1, linewidth=1, 
                                    edgecolor='black', facecolor=COMPONENT_COLORS["data_quality"], alpha=0.8)
    ax.add_patch(quality_rect)
    ax.text(2, 1.5, "Data Quality\nAssessment", ha='center', va='center', fontweight='bold')
    
    gating_rect = patches.Rectangle((5, 1), 2, 1, linewidth=1, 
                                   edgecolor='black', facecolor=COMPONENT_COLORS["gating"], alpha=0.8)
    ax.add_patch(gating_rect)
    ax.text(6, 1.5, "Gating Network", ha='center', va='center', fontweight='bold')
    
    # Draw arrow
    arrow = patches.FancyArrowPatch(
        (3, 1.5), (5, 1.5),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.5,
        color="black"
    )
    ax.add_patch(arrow)
    
    # Add step title
    ax.text(0.5, 2.5, "Step 4: Gating Network Weighting", fontsize=12, fontweight='bold')
    
    # Step 5: Integration and Final Prediction
    ax = axes[4]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Draw components
    experts_rect = patches.Rectangle((1, 1), 1.5, 1, linewidth=1, 
                                    edgecolor='black', facecolor='lightgreen', alpha=0.8)
    ax.add_patch(experts_rect)
    ax.text(1.75, 1.5, "Expert\nOutputs", ha='center', va='center', fontweight='bold')
    
    gating_rect = patches.Rectangle((1, 0.25), 1.5, 0.5, linewidth=1, 
                                   edgecolor='black', facecolor=COMPONENT_COLORS["gating"], alpha=0.8)
    ax.add_patch(gating_rect)
    ax.text(1.75, 0.5, "Weights", ha='center', va='center', fontweight='bold')
    
    integration_rect = patches.Rectangle((4, 1), 2, 1, linewidth=1, 
                                        edgecolor='black', facecolor=COMPONENT_COLORS["integration"], alpha=0.8)
    ax.add_patch(integration_rect)
    ax.text(5, 1.5, "Integration\nLayer", ha='center', va='center', fontweight='bold')
    
    output_rect = patches.Rectangle((8, 1), 1.5, 1, linewidth=1, 
                                   edgecolor='black', facecolor=COMPONENT_COLORS["output"], alpha=0.8)
    ax.add_patch(output_rect)
    ax.text(8.75, 1.5, "Final\nPrediction", ha='center', va='center', fontweight='bold')
    
    # Draw arrows
    arrow1 = patches.FancyArrowPatch(
        (2.5, 1.5), (4, 1.5),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.5,
        color="black"
    )
    ax.add_patch(arrow1)
    
    arrow2 = patches.FancyArrowPatch(
        (2.5, 0.5), (4, 1.3),
        arrowstyle="->",
        connectionstyle="arc3,rad=0.3",
        mutation_scale=15,
        linewidth=1.5,
        color="black"
    )
    ax.add_patch(arrow2)
    
    arrow3 = patches.FancyArrowPatch(
        (6, 1.5), (8, 1.5),
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.5,
        color="black"
    )
    ax.add_patch(arrow3)
    
    # Add step title
    ax.text(0.5, 2.5, "Step 5: Integration and Final Prediction", fontsize=12, fontweight='bold')
    
    # Add overall title
    plt.suptitle("MoE Framework Data Flow Sequence", fontsize=16, y=0.98)
    
    # Add description
    description = (
        "This visualization shows the step-by-step data flow through the MoE framework,\n"
        "from initial data ingestion to final prediction output."
    )
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save visualization
    output_path = os.path.join(output_dir, "data_flow_sequence.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Created data flow sequence visualization: {output_path}")
    return output_path

def create_html_summary(output_paths, output_dir):
    """Create an HTML summary of all visualizations."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MoE Architecture Flow Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            .viz-container { margin-bottom: 30px; }
            img { max-width: 100%; border: 1px solid #ddd; }
            .description { margin-top: 15px; font-style: italic; color: #666; }
        </style>
    </head>
    <body>
        <h1>MoE Architecture Flow Analysis</h1>
    """
    
    descriptions = {
        "architecture_flow.png": "Comprehensive visualization of the MoE framework architecture, showing how components are connected and how data flows through the system.",
        "data_flow_sequence.png": "Step-by-step visualization of the data flow sequence in the MoE framework, from data ingestion to final prediction."
    }
    
    for viz_path in output_paths:
        if viz_path:
            viz_name = os.path.basename(viz_path)
            viz_title = ' '.join(viz_name.split('_')).replace('.png', '').title()
            
            html_content += f"""
            <div class="viz-container">
                <h2>{viz_title}</h2>
                <img src="{viz_name}" alt="{viz_title}">
                <p class="description">{descriptions.get(viz_name, '')}</p>
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save HTML file
    html_path = os.path.join(output_dir, "architecture_flow_analysis.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path

def main():
    parser = argparse.ArgumentParser(description='Generate MoE architecture flow visualizations')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint JSON file')
    parser.add_argument('--output-dir', type=str, help='Output directory for visualizations')
    parser.add_argument('--browser', action='store_true', help='Open visualization in browser')
    args = parser.parse_args()
    
    # Find latest checkpoint if not specified
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_dir = os.path.join("results", "moe_run", "dev", "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoint_files = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.endswith('.json')],
                key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
                reverse=True
            )
            if checkpoint_files:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                logger.info(f"Found latest checkpoint: {checkpoint_path}")
    
    # Set up output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="moe_architecture_viz_")
        logger.info(f"Using temporary directory for visualizations: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint data if available
    checkpoint_data = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_data = load_checkpoint(checkpoint_path)
    
    # Generate visualizations
    logger.info("Generating architecture flow visualizations...")
    output_paths = []
    
    # Architecture flow visualization
    architecture_path = visualize_architecture_flow(checkpoint_data, output_dir)
    if architecture_path:
        output_paths.append(architecture_path)
    
    # Data flow sequence visualization
    sequence_path = visualize_data_flow_sequence(checkpoint_data, output_dir)
    if sequence_path:
        output_paths.append(sequence_path)
    
    # Create HTML summary
    html_path = create_html_summary(output_paths, output_dir)
    logger.info(f"Created visualization summary HTML: {html_path}")
    
    # Open in browser if requested
    if args.browser and html_path:
        import webbrowser
        webbrowser.open(f"file://{html_path}")
    
    logger.info(f"Architecture flow visualization complete. All files saved to: {output_dir}")
    if output_dir.startswith(tempfile.gettempdir()):
        logger.info("Note: If using a temporary directory, files will be deleted when the system is restarted.")

if __name__ == "__main__":
    main()
