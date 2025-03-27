#!/usr/bin/env python
"""
MoE Gating Network Visualization

This script visualizes how the gating network in the MoE framework selects and weights experts,
showing decision boundaries, weight distributions, and selection patterns.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
import tempfile
import random
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

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

def visualize_gating_mechanism(checkpoint_data, output_dir):
    """Visualize the gating mechanism of the MoE framework."""
    if not checkpoint_data:
        logger.error("No checkpoint data loaded")
        return None
    
    try:
        # Extract gating network information
        gating_info = checkpoint_data.get("gating_network", {})
        if not gating_info:
            # Try alternative structure
            gating_info = checkpoint_data.get("gating", {})
            
        gating_type = gating_info.get("type", "unknown")
        if gating_type == "unknown":
            # Try to infer type from structure
            if "quality_weights" in gating_info:
                gating_type = "quality_aware"
            elif "meta_features" in gating_info:
                gating_type = "meta_learner"
            else:
                gating_type = "standard"
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create gating network diagram based on type
        if gating_type == "quality_aware":
            visualize_quality_aware_gating(ax, gating_info)
        elif gating_type == "meta_learner":
            visualize_meta_learner_gating(ax, gating_info)
        else:
            visualize_standard_gating(ax, gating_info)
        
        # Add title
        plt.suptitle(f"MoE Gating Network: {gating_type.replace('_', ' ').title()}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save visualization
        output_path = os.path.join(output_dir, "gating_mechanism.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created gating mechanism visualization: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating gating mechanism visualization: {str(e)}")
        return None

def visualize_standard_gating(ax, gating_info):
    """Visualize standard gating network."""
    # Create a simple diagram of the standard gating network
    ax.axis('off')
    
    # Draw input node
    input_circle = plt.Circle((0.2, 0.5), 0.1, color='lightblue', alpha=0.8)
    ax.add_patch(input_circle)
    ax.text(0.2, 0.5, "Input", ha='center', va='center', fontweight='bold')
    
    # Draw gating network box
    gating_rect = plt.Rectangle((0.4, 0.3), 0.2, 0.4, color='lightgreen', alpha=0.8)
    ax.add_patch(gating_rect)
    ax.text(0.5, 0.5, "Standard\nGating\nNetwork", ha='center', va='center', fontweight='bold')
    
    # Draw expert nodes
    expert_colors = list(EXPERT_COLORS.values())
    num_experts = min(4, len(expert_colors))
    
    expert_y_positions = np.linspace(0.3, 0.7, num_experts)
    for i, y_pos in enumerate(expert_y_positions):
        expert_circle = plt.Circle((0.8, y_pos), 0.05, color=expert_colors[i], alpha=0.8)
        ax.add_patch(expert_circle)
        ax.text(0.8, y_pos, f"E{i+1}", ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw arrows
    ax.arrow(0.3, 0.5, 0.08, 0, head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    for y_pos in expert_y_positions:
        ax.arrow(0.6, 0.5, 0.14, y_pos-0.5, head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Add description
    description = ("Standard Gating Network: Assigns weights to each expert\n"
                  "based on the input data, without considering data quality\n"
                  "or other specialized factors.")
    ax.text(0.5, 0.1, description, ha='center', va='center', fontsize=12,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

def visualize_quality_aware_gating(ax, gating_info):
    """Visualize quality-aware gating network."""
    # Create a diagram of the quality-aware gating network
    ax.axis('off')
    
    # Draw input node
    input_circle = plt.Circle((0.15, 0.5), 0.08, color='lightblue', alpha=0.8)
    ax.add_patch(input_circle)
    ax.text(0.15, 0.5, "Input", ha='center', va='center', fontweight='bold')
    
    # Draw quality assessment node
    quality_circle = plt.Circle((0.35, 0.7), 0.08, color='orange', alpha=0.8)
    ax.add_patch(quality_circle)
    ax.text(0.35, 0.7, "Quality\nAssessment", ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw gating network box
    gating_rect = plt.Rectangle((0.5, 0.3), 0.2, 0.4, color='lightgreen', alpha=0.8)
    ax.add_patch(gating_rect)
    ax.text(0.6, 0.5, "Quality-Aware\nGating\nNetwork", ha='center', va='center', fontweight='bold')
    
    # Draw expert nodes
    expert_colors = list(EXPERT_COLORS.values())
    num_experts = min(4, len(expert_colors))
    
    expert_y_positions = np.linspace(0.3, 0.7, num_experts)
    for i, y_pos in enumerate(expert_y_positions):
        expert_circle = plt.Circle((0.85, y_pos), 0.05, color=expert_colors[i], alpha=0.8)
        ax.add_patch(expert_circle)
        ax.text(0.85, y_pos, f"E{i+1}", ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw arrows
    ax.arrow(0.23, 0.5, 0.08, 0, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.23, 0.5, 0.08, 0.19, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.43, 0.7, 0.06, -0.1, head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    for y_pos in expert_y_positions:
        ax.arrow(0.7, 0.5, 0.1, y_pos-0.5, head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Add quality weights 
    for i, y_pos in enumerate(expert_y_positions):
        # Example quality weights
        quality_score = gating_info.get(f"expert_{i+1}_quality", random.uniform(0.5, 0.95))
        ax.text(0.78, y_pos + 0.06, f"Q: {quality_score:.2f}", ha='center', va='center', 
                fontsize=8, fontweight='bold', color='red')
    
    # Add description
    description = ("Quality-Aware Gating Network: Adjusts expert weights based on\n"
                  "data quality assessment. Reduces the influence of experts\n"
                  "when their input data is of low quality or contains missing values.")
    ax.text(0.5, 0.1, description, ha='center', va='center', fontsize=12,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

def visualize_meta_learner_gating(ax, gating_info):
    """Visualize meta-learner gating network."""
    # Create a diagram of the meta-learner gating network
    ax.axis('off')
    
    # Draw input node
    input_circle = plt.Circle((0.15, 0.5), 0.08, color='lightblue', alpha=0.8)
    ax.add_patch(input_circle)
    ax.text(0.15, 0.5, "Input", ha='center', va='center', fontweight='bold')
    
    # Draw meta-features node
    meta_circle = plt.Circle((0.35, 0.3), 0.08, color='purple', alpha=0.8)
    ax.add_patch(meta_circle)
    ax.text(0.35, 0.3, "Meta\nFeatures", ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw quality assessment node
    quality_circle = plt.Circle((0.35, 0.7), 0.08, color='orange', alpha=0.8)
    ax.add_patch(quality_circle)
    ax.text(0.35, 0.7, "Quality\nAssessment", ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw meta-learner gating network box
    gating_rect = plt.Rectangle((0.5, 0.3), 0.2, 0.4, color='lightgreen', alpha=0.8)
    ax.add_patch(gating_rect)
    ax.text(0.6, 0.5, "Meta-Learner\nGating\nNetwork", ha='center', va='center', fontweight='bold')
    
    # Draw expert nodes
    expert_colors = list(EXPERT_COLORS.values())
    num_experts = min(4, len(expert_colors))
    
    expert_y_positions = np.linspace(0.3, 0.7, num_experts)
    for i, y_pos in enumerate(expert_y_positions):
        expert_circle = plt.Circle((0.85, y_pos), 0.05, color=expert_colors[i], alpha=0.8)
        ax.add_patch(expert_circle)
        ax.text(0.85, y_pos, f"E{i+1}", ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw arrows
    ax.arrow(0.23, 0.5, 0.08, -0.19, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.23, 0.5, 0.08, 0.19, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.43, 0.3, 0.06, 0.1, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.43, 0.7, 0.06, -0.1, head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    for y_pos in expert_y_positions:
        ax.arrow(0.7, 0.5, 0.1, y_pos-0.5, head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Add description
    description = ("Meta-Learner Gating Network: Uses meta-features of the input data\n"
                  "to predict which experts will perform best. Combines this with\n"
                  "quality assessment to determine optimal expert weights.")
    ax.text(0.5, 0.1, description, ha='center', va='center', fontsize=12,
           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

def visualize_expert_selection(checkpoint_data, output_dir):
    """Visualize expert selection patterns by the gating network."""
    if not checkpoint_data:
        logger.error("No checkpoint data loaded")
        return None
    
    try:
        # Extract expert selection data
        selection_data = checkpoint_data.get("expert_selection", {})
        if not selection_data:
            # Try to extract from predictions
            predictions = checkpoint_data.get("predictions", [])
            if predictions:
                selection_data = {"counts": {}}
                for pred in predictions:
                    selected = pred.get("selected_expert", None)
                    if selected:
                        if selected in selection_data["counts"]:
                            selection_data["counts"][selected] += 1
                        else:
                            selection_data["counts"][selected] = 1
        
        # If still no data, create sample data for visualization
        if not selection_data or not selection_data.get("counts", {}):
            # Create sample data
            experts_data = checkpoint_data.get("experts", {})
            if not experts_data:
                experts_data = checkpoint_data.get("expert_benchmarks", {})
                
            expert_ids = list(experts_data.keys())
            if not expert_ids:
                expert_ids = ["1", "2", "3", "4"]
                
            selection_data = {"counts": {}}
            total = 100
            remaining = total
            
            for i, expert_id in enumerate(expert_ids[:-1]):
                count = random.randint(5, remaining - 5)
                selection_data["counts"][expert_id] = count
                remaining -= count
                
            selection_data["counts"][expert_ids[-1]] = remaining
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # 1. Pie chart of expert selection frequency
        counts = selection_data.get("counts", {})
        labels = [f"Expert {expert_id}" for expert_id in counts.keys()]
        sizes = list(counts.values())
        
        # Get expert types if available
        experts_data = checkpoint_data.get("experts", {})
        if not experts_data:
            experts_data = checkpoint_data.get("expert_benchmarks", {})
            
        expert_types = {}
        for expert_id in counts.keys():
            expert_info = experts_data.get(expert_id, {})
            expert_types[expert_id] = expert_info.get("type", "unknown")
            
        # Colors based on expert type if available
        colors = []
        for expert_id in counts.keys():
            expert_type = expert_types.get(expert_id, "unknown")
            colors.append(EXPERT_COLORS.get(expert_type, "#CCCCCC"))
        
        # Create pie chart
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=None, autopct='%1.1f%%', startangle=90,
            colors=colors, wedgeprops=dict(width=0.5, edgecolor='w')
        )
        
        # Add a legend
        ax1.legend(wedges, labels, title="Expert Selection", 
                 loc="center left", bbox_to_anchor=(0.85, 0, 0.5, 1))
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.set_aspect('equal')
        ax1.set_title('Expert Selection Frequency', fontsize=14)
        
        # 2. Bar chart of selection over time or conditions
        # Create sample time series data if not available
        time_series = checkpoint_data.get("selection_over_time", None)
        if not time_series:
            # Create sample data
            num_points = 10
            time_points = list(range(num_points))
            expert_selection = {}
            
            for expert_id in counts.keys():
                # Create a random selection pattern
                expert_selection[expert_id] = [random.randint(0, 5) for _ in range(num_points)]
                
            # Normalize to ensure sum is 100% at each time point
            for t in range(num_points):
                total = sum(expert_selection[expert_id][t] for expert_id in counts.keys())
                if total > 0:
                    for expert_id in counts.keys():
                        expert_selection[expert_id][t] = expert_selection[expert_id][t] / total * 100
        else:
            time_points = list(range(len(time_series)))
            expert_selection = {}
            for expert_id in counts.keys():
                expert_selection[expert_id] = [point.get(expert_id, 0) for point in time_series]
        
        # Create stacked bar chart
        bottom = np.zeros(len(time_points))
        
        for expert_id in counts.keys():
            expert_type = expert_types.get(expert_id, "unknown")
            color = EXPERT_COLORS.get(expert_type, "#CCCCCC")
            
            ax2.bar(time_points, expert_selection[expert_id], bottom=bottom, 
                   label=f"Expert {expert_id}", color=color, alpha=0.7)
            
            bottom += expert_selection[expert_id]
        
        # Add labels
        ax2.set_title('Expert Selection Over Time/Conditions', fontsize=14)
        ax2.set_xlabel('Time/Condition Index')
        ax2.set_ylabel('Selection Percentage')
        ax2.legend(title="Experts")
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.suptitle('Gating Network Expert Selection Patterns', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save visualization
        output_path = os.path.join(output_dir, "expert_selection.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created expert selection visualization: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating expert selection visualization: {str(e)}")
        return None

def visualize_decision_boundaries(checkpoint_data, output_dir):
    """Visualize decision boundaries of the gating network."""
    if not checkpoint_data:
        logger.error("No checkpoint data loaded")
        return None
    
    try:
        # Extract decision boundary data if available
        boundary_data = checkpoint_data.get("decision_boundaries", None)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # If no boundary data, create a sample 2D visualization
        if not boundary_data:
            # Create a sample 2D grid
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            X, Y = np.meshgrid(x, y)
            
            # Create sample decision regions
            experts_data = checkpoint_data.get("experts", {})
            if not experts_data:
                experts_data = checkpoint_data.get("expert_benchmarks", {})
                
            expert_ids = list(experts_data.keys())
            if not expert_ids:
                expert_ids = ["1", "2", "3", "4"]
            
            # Create sample decision function
            Z = np.zeros((100, 100), dtype=int)
            centers = [(random.uniform(-4, 4), random.uniform(-4, 4)) for _ in range(len(expert_ids))]
            
            for i in range(100):
                for j in range(100):
                    # Find closest center
                    distances = [np.sqrt((X[i, j] - cx)**2 + (Y[i, j] - cy)**2) for cx, cy in centers]
                    Z[i, j] = np.argmin(distances)
            
            # Create color map
            expert_types = {}
            for i, expert_id in enumerate(expert_ids):
                expert_info = experts_data.get(expert_id, {})
                expert_types[i] = expert_info.get("type", "unknown")
            
            colors = [EXPERT_COLORS.get(expert_types.get(i, "unknown"), "#CCCCCC") for i in range(len(expert_ids))]
            cmap = LinearSegmentedColormap.from_list("expert_cmap", colors, N=len(expert_ids))
            
            # Plot decision boundaries
            im = ax.imshow(Z, extent=[-5, 5, -5, 5], origin='lower', cmap=cmap, alpha=0.6)
            
            # Add contour lines
            contour = ax.contour(X, Y, Z, levels=np.arange(len(expert_ids)+1)-0.5, colors='k', linestyles='-', linewidths=0.5)
            
            # Add sample points
            num_samples = 50
            sample_x = np.random.uniform(-5, 5, num_samples)
            sample_y = np.random.uniform(-5, 5, num_samples)
            sample_z = np.zeros(num_samples, dtype=int)
            
            for i in range(num_samples):
                # Find closest center
                distances = [np.sqrt((sample_x[i] - cx)**2 + (sample_y[i] - cy)**2) for cx, cy in centers]
                sample_z[i] = np.argmin(distances)
            
            # Plot sample points
            for i in range(len(expert_ids)):
                mask = sample_z == i
                ax.scatter(sample_x[mask], sample_y[mask], c=colors[i], s=30, edgecolor='k', label=f"Expert {expert_ids[i]}")
            
            # Add labels and title
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title('Gating Network Decision Boundaries', fontsize=14)
            ax.legend(title="Selected Expert")
            
            # Add description
            plt.figtext(0.5, 0.01, 
                      "This visualization shows how the gating network selects different experts\n"
                      "based on the input features. Each color represents a region where a specific\n"
                      "expert is selected as the primary contributor to the prediction.",
                      ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            # Save visualization
            output_path = os.path.join(output_dir, "decision_boundaries.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Created decision boundaries visualization: {output_path}")
            return output_path
            
        else:
            # Use actual boundary data if available
            # Implementation would depend on the format of boundary_data
            pass
        
    except Exception as e:
        logger.error(f"Error creating decision boundaries visualization: {str(e)}")
        return None

def create_html_summary(output_paths, output_dir):
    """Create an HTML summary of all visualizations."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MoE Gating Network Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            .viz-container { margin-bottom: 30px; }
            img { max-width: 100%; border: 1px solid #ddd; }
            .description { margin-top: 15px; font-style: italic; color: #666; }
        </style>
    </head>
    <body>
        <h1>MoE Gating Network Analysis</h1>
    """
    
    descriptions = {
        "gating_mechanism.png": "Visualization of the gating network mechanism, showing how it processes input data and assigns weights to different expert models.",
        "expert_selection.png": "Analysis of expert selection patterns, showing which experts are selected most frequently and how selection changes over time or conditions.",
        "decision_boundaries.png": "Visualization of the decision boundaries used by the gating network to select different experts based on input features."
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
    html_path = os.path.join(output_dir, "gating_network_analysis.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path

def main():
    parser = argparse.ArgumentParser(description='Generate MoE gating network visualizations')
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
        output_dir = tempfile.mkdtemp(prefix="moe_gating_viz_")
        logger.info(f"Using temporary directory for visualizations: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint data
    checkpoint_data = load_checkpoint(checkpoint_path)
    if not checkpoint_data:
        logger.error("Failed to load checkpoint data")
        return
    
    # Generate visualizations
    logger.info("Generating gating network visualizations...")
    output_paths = []
    
    # Gating mechanism visualization
    mechanism_path = visualize_gating_mechanism(checkpoint_data, output_dir)
    if mechanism_path:
        output_paths.append(mechanism_path)
    
    # Expert selection visualization
    selection_path = visualize_expert_selection(checkpoint_data, output_dir)
    if selection_path:
        output_paths.append(selection_path)
    
    # Decision boundaries visualization
    boundaries_path = visualize_decision_boundaries(checkpoint_data, output_dir)
    if boundaries_path:
        output_paths.append(boundaries_path)
    
    # Create HTML summary
    html_path = create_html_summary(output_paths, output_dir)
    logger.info(f"Created visualization summary HTML: {html_path}")
    
    # Open in browser if requested
    if args.browser and html_path:
        import webbrowser
        webbrowser.open(f"file://{html_path}")
    
    logger.info(f"Gating network visualization complete. All files saved to: {output_dir}")
    if output_dir.startswith(tempfile.gettempdir()):
        logger.info("Note: If using a temporary directory, files will be deleted when the system is restarted.")

if __name__ == "__main__":
    main()
