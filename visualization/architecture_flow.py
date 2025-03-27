#!/usr/bin/env python
"""
MoE Architecture Flow Visualization

This script creates visualizations that accurately represent the MoE framework architecture,
showing how data flows through the system and how components interact.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
from pathlib import Path
import argparse
import tempfile

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

def visualize_architecture_diagram(checkpoint_data, output_dir):
    """Create a comprehensive architecture diagram of the MoE framework."""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Define component coordinates (x, y)
        components = {
            # Data layer
            'data_input': (0.1, 0.8),
            'preprocessing': (0.3, 0.8),
            
            # Expert layer
            'physiological_expert': (0.2, 0.6),
            'environmental_expert': (0.4, 0.6),
            'behavioral_expert': (0.6, 0.6),
            'medication_expert': (0.8, 0.6),
            
            # Gating layer
            'gating_network': (0.5, 0.4),
            
            # Integration layer
            'integration_connector': (0.5, 0.3),
            
            # Output layer
            'prediction_output': (0.5, 0.1),
            
            # Management components
            'state_manager': (0.9, 0.8),
            'event_system': (0.9, 0.5),
            'meta_learner': (0.2, 0.4)
        }
        
        # Draw components
        # Data layer components
        draw_component(ax, components['data_input'], 0.1, 0.05, 'Data Input', 'lightblue')
        draw_component(ax, components['preprocessing'], 0.15, 0.05, 'Preprocessing\nManager', 'lightblue')
        
        # Expert layer components
        draw_component(ax, components['physiological_expert'], 0.1, 0.08, 'Physiological\nExpert', EXPERT_COLORS['physiological'])
        draw_component(ax, components['environmental_expert'], 0.1, 0.08, 'Environmental\nExpert', EXPERT_COLORS['environmental'])
        draw_component(ax, components['behavioral_expert'], 0.1, 0.08, 'Behavioral\nExpert', EXPERT_COLORS['behavioral'])
        draw_component(ax, components['medication_expert'], 0.1, 0.08, 'Medication\nExpert', EXPERT_COLORS['medication_history'])
        
        # Gating network
        draw_component(ax, components['gating_network'], 0.2, 0.08, 'Gating Network', 'lightgreen')
        
        # Integration layer
        draw_component(ax, components['integration_connector'], 0.2, 0.05, 'Integration Connector', 'orange')
        
        # Output layer
        draw_component(ax, components['prediction_output'], 0.1, 0.05, 'Prediction Output', 'lightgray')
        
        # Management components
        draw_component(ax, components['state_manager'], 0.1, 0.05, 'State Manager', 'lightgray')
        draw_component(ax, components['event_system'], 0.1, 0.05, 'Event System', 'lightgray')
        draw_component(ax, components['meta_learner'], 0.1, 0.05, 'Meta Learner', 'purple')
        
        # Draw connections
        # Data flow connections
        draw_arrow(ax, components['data_input'], components['preprocessing'])
        draw_arrow(ax, components['preprocessing'], components['physiological_expert'])
        draw_arrow(ax, components['preprocessing'], components['environmental_expert'])
        draw_arrow(ax, components['preprocessing'], components['behavioral_expert'])
        draw_arrow(ax, components['preprocessing'], components['medication_expert'])
        
        # Expert to gating connections
        draw_arrow(ax, components['physiological_expert'], components['gating_network'])
        draw_arrow(ax, components['environmental_expert'], components['gating_network'])
        draw_arrow(ax, components['behavioral_expert'], components['gating_network'])
        draw_arrow(ax, components['medication_expert'], components['gating_network'])
        
        # Meta-learner connection
        draw_arrow(ax, components['meta_learner'], components['gating_network'])
        
        # Gating to integration
        draw_arrow(ax, components['gating_network'], components['integration_connector'])
        
        # Integration to output
        draw_arrow(ax, components['integration_connector'], components['prediction_output'])
        
        # State manager connections
        draw_dashed_arrow(ax, components['state_manager'], components['gating_network'])
        draw_dashed_arrow(ax, components['state_manager'], components['integration_connector'])
        
        # Event system connections
        draw_dashed_arrow(ax, components['event_system'], components['physiological_expert'])
        draw_dashed_arrow(ax, components['event_system'], components['environmental_expert'])
        draw_dashed_arrow(ax, components['event_system'], components['behavioral_expert'])
        draw_dashed_arrow(ax, components['event_system'], components['medication_expert'])
        draw_dashed_arrow(ax, components['event_system'], components['gating_network'])
        
        # Add title and legend
        plt.suptitle("MoE Framework Architecture", fontsize=18, y=0.98)
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Data Layer'),
            mpatches.Patch(color=EXPERT_COLORS['physiological'], label='Physiological Expert'),
            mpatches.Patch(color=EXPERT_COLORS['environmental'], label='Environmental Expert'),
            mpatches.Patch(color=EXPERT_COLORS['behavioral'], label='Behavioral Expert'),
            mpatches.Patch(color=EXPERT_COLORS['medication_history'], label='Medication Expert'),
            mpatches.Patch(color='lightgreen', label='Gating Network'),
            mpatches.Patch(color='orange', label='Integration Layer'),
            mpatches.Patch(color='lightgray', label='Support Components'),
            mpatches.Patch(color='purple', label='Meta-Learner')
        ]
        
        # Add solid/dashed line to legend
        solid_line = plt.Line2D([0], [0], color='black', lw=1, label='Data Flow')
        dashed_line = plt.Line2D([0], [0], color='black', lw=1, linestyle='--', label='Control Flow')
        legend_elements.extend([solid_line, dashed_line])
        
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                 ncol=3, fontsize=10)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save visualization
        output_path = os.path.join(output_dir, "architecture_diagram.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created architecture diagram: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating architecture diagram: {str(e)}")
        return None

def visualize_data_flow(checkpoint_data, output_dir):
    """Visualize the data flow through the MoE pipeline."""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        
        # Define process steps coordinates (x, y)
        steps = [
            (0.1, 0.5, "Data\nInput"),
            (0.25, 0.5, "Preprocessing"),
            (0.4, 0.5, "Expert\nModels"),
            (0.55, 0.5, "Gating\nNetwork"),
            (0.7, 0.5, "Integration"),
            (0.85, 0.5, "Prediction\nOutput")
        ]
        
        # Draw main flow
        for i in range(len(steps)):
            # Draw step circle
            circle = plt.Circle((steps[i][0], steps[i][1]), 0.05, 
                               facecolor='lightblue', edgecolor='black', alpha=0.8)
            ax.add_patch(circle)
            
            # Add step label
            ax.text(steps[i][0], steps[i][1], f"{i+1}", ha='center', va='center', 
                   fontweight='bold', fontsize=12)
            
            # Add step name below
            ax.text(steps[i][0], steps[i][1] - 0.1, steps[i][2], ha='center', va='center')
            
            # Add arrow to next step
            if i < len(steps) - 1:
                ax.arrow(steps[i][0] + 0.05, steps[i][1], 
                        steps[i+1][0] - steps[i][0] - 0.1, 0, 
                        head_width=0.02, head_length=0.01, fc='black', ec='black')
        
        # Add expert details
        expert_types = ["Physiological", "Environmental", "Behavioral", "Medication"]
        expert_y = [0.7, 0.6, 0.4, 0.3]
        
        for i, expert_type in enumerate(expert_types):
            # Draw expert box
            rect = plt.Rectangle((0.4 - 0.05, expert_y[i] - 0.03), 0.1, 0.06, 
                               facecolor=EXPERT_COLORS.get(expert_type.lower(), '#CCCCCC'),
                               alpha=0.8, edgecolor='black')
            ax.add_patch(rect)
            
            # Add expert label
            ax.text(0.4, expert_y[i], expert_type, ha='center', va='center', fontsize=10)
            
            # Add arrow from preprocessing to expert
            ax.arrow(0.25, 0.5, 0.1, expert_y[i] - 0.5, head_width=0.01, 
                    head_length=0.01, fc='black', ec='black', linestyle='-')
            
            # Add arrow from expert to gating
            ax.arrow(0.45, expert_y[i], 0.05, 0.5 - expert_y[i], head_width=0.01, 
                    head_length=0.01, fc='black', ec='black', linestyle='-')
        
        # Add information boxes for key steps
        info_boxes = [
            (0.25, 0.8, "Preprocessing", 
             "- Data cleaning\n- Feature extraction\n- Domain separation\n- Quality assessment"),
            (0.55, 0.8, "Gating Network", 
             "- Expert weighting\n- Quality-aware selection\n- Meta-learning integration\n- Adaptive weighting"),
            (0.7, 0.8, "Integration", 
             "- Weighted combination\n- Confidence assessment\n- Performance tracking\n- Explainability")
        ]
        
        for x, y, title, content in info_boxes:
            # Draw info box
            rect = plt.Rectangle((x - 0.08, y - 0.15), 0.16, 0.3, 
                               facecolor='lightyellow', alpha=0.8, 
                               edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add title and content
            ax.text(x, y + 0.1, title, ha='center', va='center', 
                   fontweight='bold', fontsize=12)
            ax.text(x, y - 0.05, content, ha='center', va='center', 
                   fontsize=9, linespacing=1.5)
            
            # Add connector line
            ax.plot([x, x], [y - 0.15, 0.55], 'k--', alpha=0.5)
        
        # Add title
        plt.suptitle("MoE Pipeline Data Flow", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save visualization
        output_path = os.path.join(output_dir, "data_flow.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created data flow visualization: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating data flow visualization: {str(e)}")
        return None

def visualize_expert_structure(checkpoint_data, output_dir):
    """Visualize the expert model structure in the MoE framework."""
    if not checkpoint_data:
        logger.error("No checkpoint data loaded")
        return None
    
    # Extract expert model information
    try:
        experts_data = checkpoint_data.get("experts", {})
        if not experts_data:
            experts_data = {}
            # Try alternative structure
            expert_benchmarks = checkpoint_data.get("expert_benchmarks", {})
            if expert_benchmarks:
                for expert_id, data in expert_benchmarks.items():
                    experts_data[expert_id] = {"type": data.get("type", "unknown")}
        
        expert_ids = list(experts_data.keys())
        
        if not expert_ids:
            logger.warning("No expert data found in checkpoint")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create expert structure diagram
        expert_types = []
        for expert_id in expert_ids:
            expert_info = experts_data.get(expert_id, {})
            expert_type = expert_info.get("type", "unknown")
            expert_types.append(expert_type)
        
        # Count expert types
        expert_type_counts = {}
        for expert_type in expert_types:
            if expert_type in expert_type_counts:
                expert_type_counts[expert_type] += 1
            else:
                expert_type_counts[expert_type] = 1
        
        # Create horizontal bar chart of expert types
        types = list(expert_type_counts.keys())
        counts = list(expert_type_counts.values())
        colors = [EXPERT_COLORS.get(t, "#CCCCCC") for t in types]
        
        bars = ax.barh(types, counts, color=colors)
        
        # Add expert count labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f"{counts[i]} expert{'s' if counts[i] > 1 else ''}",
                    va='center')
        
        # Add titles and labels
        ax.set_title("MoE Expert Model Structure", fontsize=16, pad=20)
        ax.set_xlabel("Number of Experts", fontsize=12)
        ax.set_ylabel("Expert Domain", fontsize=12)
        ax.set_xlim(0, max(counts) + 2)
        
        # Add descriptive annotations
        expert_descriptions = {
            "physiological": "Biomarker and vital sign analysis",
            "environmental": "External factors and triggers",
            "behavioral": "Activity patterns and lifestyle",
            "medication_history": "Medication efficacy and timing"
        }
        
        # Add description text box
        descriptions = [f"{t}: {expert_descriptions.get(t, '')}" for t in types]
        description_text = "\n".join(descriptions)
        
        plt.figtext(0.15, 0.01, description_text, 
                  wrap=True, horizontalalignment='left', 
                  fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(output_dir, "expert_structure.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created expert structure visualization: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating expert structure visualization: {str(e)}")
        return None

def draw_component(ax, position, width, height, label, color):
    """Helper to draw a component box with label."""
    rect = plt.Rectangle((position[0] - width/2, position[1] - height/2), 
                          width, height, facecolor=color, alpha=0.8, 
                          edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(position[0], position[1], label, ha='center', va='center', 
           fontsize=10, fontweight='bold')

def draw_arrow(ax, start_pos, end_pos):
    """Helper to draw an arrow between components."""
    ax.arrow(start_pos[0], start_pos[1], 
            end_pos[0] - start_pos[0], end_pos[1] - start_pos[1],
            head_width=0.01, head_length=0.01, fc='black', ec='black',
            length_includes_head=True)

def draw_dashed_arrow(ax, start_pos, end_pos):
    """Helper to draw a dashed arrow between components."""
    ax.annotate("", xy=end_pos, xytext=start_pos,
               arrowprops=dict(arrowstyle="->", linestyle='dashed'))

def create_html_summary(output_paths, output_dir):
    """Create an HTML summary of all visualizations."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MoE Architecture Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            .viz-container { margin-bottom: 30px; }
            img { max-width: 100%; border: 1px solid #ddd; }
            .description { margin-top: 15px; font-style: italic; color: #666; }
        </style>
    </head>
    <body>
        <h1>MoE Architecture Visualization</h1>
    """
    
    descriptions = {
        "architecture_diagram.png": "Comprehensive diagram showing the MoE framework architecture with all components and their connections.",
        "data_flow.png": "Visualization of data flow through the MoE pipeline, from input to prediction output.",
        "expert_structure.png": "Analysis of expert model structure, showing the distribution of expert types in the framework."
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
    html_path = os.path.join(output_dir, "architecture_visualization.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path

def main():
    parser = argparse.ArgumentParser(description='Generate MoE architecture visualizations')
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
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.error("No valid checkpoint file found")
        return
    
    # Set up output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="moe_arch_viz_")
        logger.info(f"Using temporary directory for visualizations: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint data
    checkpoint_data = load_checkpoint(checkpoint_path)
    if not checkpoint_data:
        logger.error("Failed to load checkpoint data")
        return
    
    # Generate visualizations
    logger.info("Generating architecture visualizations...")
    output_paths = []
    
    # Architecture diagram
    arch_path = visualize_architecture_diagram(checkpoint_data, output_dir)
    if arch_path:
        output_paths.append(arch_path)
    
    # Data flow diagram
    flow_path = visualize_data_flow(checkpoint_data, output_dir)
    if flow_path:
        output_paths.append(flow_path)
    
    # Expert structure visualization
    expert_path = visualize_expert_structure(checkpoint_data, output_dir)
    if expert_path:
        output_paths.append(expert_path)
    
    # Create HTML summary
    html_path = create_html_summary(output_paths, output_dir)
    logger.info(f"Created visualization summary HTML: {html_path}")
    
    # Open in browser if requested
    if args.browser and html_path:
        import webbrowser
        webbrowser.open(f"file://{html_path}")
    
    logger.info(f"Architecture visualization complete. All files saved to: {output_dir}")
    if output_dir.startswith(tempfile.gettempdir()):
        logger.info("Note: If using a temporary directory, files will be deleted when the system is restarted.")

if __name__ == "__main__":
    main()
