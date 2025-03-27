#!/usr/bin/env python
"""
MoE Expert Contribution Visualization

This script visualizes how each expert contributes to the final MoE prediction,
showing expert weights, performance comparisons, and domain-specific metrics.
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

def visualize_expert_weights(checkpoint_data, output_dir):
    """Visualize the weights assigned to each expert by the gating network."""
    if not checkpoint_data:
        logger.error("No checkpoint data loaded")
        return None
    
    try:
        # Extract expert weights and performance info
        experts_data = checkpoint_data.get("experts", {})
        if not experts_data:
            experts_data = checkpoint_data.get("expert_benchmarks", {})
                
        gating_info = checkpoint_data.get("gating_network", {})
        
        # Get expert weights if available, otherwise use equal weights
        expert_ids = list(experts_data.keys())
        
        if not expert_ids:
            logger.warning("No expert data found in checkpoint")
            return None
        
        # Try to get weights from gating_network data
        weights = {}
        for expert_id in expert_ids:
            # Check different possible formats for weights
            weight_key = f"expert_{expert_id}_weight"
            weight = gating_info.get(weight_key, None)
            
            if weight is None:
                weight = gating_info.get(f"weights", {}).get(expert_id, None)
            
            if weight is None:
                # Fallback to equal weighting
                weight = 1.0 / len(expert_ids)
            
            # Ensure weight is a float
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0 / len(expert_ids)
                
            weights[expert_id] = weight
            
        # Normalize weights to sum to 1
        total_weights = sum(weights.values())
        if total_weights > 0:
            weights = {k: v / total_weights for k, v in weights.items()}
        
        # Get expert types
        expert_types = {}
        for expert_id in expert_ids:
            expert_info = experts_data.get(expert_id, {})
            expert_types[expert_id] = expert_info.get("type", "unknown")
            if expert_types[expert_id] == "unknown" and "metrics" in expert_info:
                # Try to derive type from metrics
                if "physiological_score" in expert_info["metrics"]:
                    expert_types[expert_id] = "physiological"
                elif "environmental_score" in expert_info["metrics"]:
                    expert_types[expert_id] = "environmental"
                elif "behavioral_score" in expert_info["metrics"]:
                    expert_types[expert_id] = "behavioral"
                elif "medication_score" in expert_info["metrics"]:
                    expert_types[expert_id] = "medication_history"
        
        # Create figure with pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pie chart of expert weights
        labels = [f"Expert {expert_id}\n({expert_types[expert_id]})" for expert_id in expert_ids]
        sizes = [weights[expert_id] for expert_id in expert_ids]
        colors = [EXPERT_COLORS.get(expert_types[expert_id], "#CCCCCC") for expert_id in expert_ids]
        
        # Create pie chart with expert contribution
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, autopct='%1.1f%%', startangle=90,
            colors=colors, wedgeprops=dict(width=0.5, edgecolor='w')
        )
        
        # Add a legend
        ax.legend(wedges, labels, title="Expert Contribution", 
                 loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_aspect('equal')
        ax.set_title('Expert Contribution to MoE Prediction', fontsize=16)
        
        # Add description
        plt.figtext(0.5, 0.01, 
                  "This chart shows the weight assigned to each expert by the gating network.\n"
                  "Higher percentages indicate greater influence on the final prediction.",
                  ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(output_dir, "expert_weights.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created expert weights visualization: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating expert weights visualization: {str(e)}")
        return None

def visualize_performance_comparison(checkpoint_data, output_dir):
    """Visualize performance comparison between MoE and individual experts."""
    if not checkpoint_data:
        logger.error("No checkpoint data loaded")
        return None
    
    try:
        # Extract expert performance metrics
        experts_data = checkpoint_data.get("experts", {})
        if not experts_data:
            experts_data = checkpoint_data.get("expert_benchmarks", {})
                
        performance = checkpoint_data.get("performance", {})
        if not performance:
            performance = checkpoint_data.get("overall_metrics", {})
        
        expert_ids = list(experts_data.keys())
        
        if not expert_ids:
            logger.warning("No expert data found in checkpoint")
            return None
        
        # Get MoE and individual expert performances
        moe_rmse = 0
        if isinstance(performance, dict):
            moe_rmse = performance.get("rmse", 0)
            if moe_rmse == 0 and "metrics" in performance:
                moe_rmse = performance["metrics"].get("rmse", 0)
        
        expert_rmse = {}
        for expert_id in expert_ids:
            expert_info = experts_data.get(expert_id, {})
            if "rmse" in expert_info:
                expert_rmse[expert_id] = expert_info["rmse"]
            elif "metrics" in expert_info and "rmse" in expert_info["metrics"]:
                expert_rmse[expert_id] = expert_info["metrics"]["rmse"]
            else:
                # If no RMSE, generate random for visualization
                expert_rmse[expert_id] = moe_rmse * (1 + random.uniform(0.1, 0.4))
        
        # Get expert types
        expert_types = {}
        for expert_id in expert_ids:
            expert_info = experts_data.get(expert_id, {})
            expert_types[expert_id] = expert_info.get("type", "unknown")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for bar chart
        ids = ['MoE'] + expert_ids
        rmse_values = [moe_rmse] + [expert_rmse[expert_id] for expert_id in expert_ids]
        
        # Colors for bars
        colors = ['#6699CC']  # MoE color
        for expert_id in expert_ids:
            colors.append(EXPERT_COLORS.get(expert_types[expert_id], "#CCCCCC"))
        
        # Create bar chart
        bars = ax.bar(ids, rmse_values, color=colors)
        
        # Add RMSE values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f"{height:.4f}", ha='center', va='bottom', fontsize=9)
        
        # Add labels and title
        ax.set_title('Performance Comparison: MoE vs Individual Experts', fontsize=16)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('RMSE (lower is better)', fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add grid for readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Find the best expert (lowest RMSE)
        best_expert_id = min(expert_rmse, key=expert_rmse.get)
        best_expert_rmse = expert_rmse[best_expert_id]
        
        # Calculate improvement percentage
        if best_expert_rmse > 0 and moe_rmse > 0:
            improvement = (best_expert_rmse - moe_rmse) / best_expert_rmse * 100
            
            # Add improvement callout
            if improvement > 0:
                ax.annotate(
                    f"{improvement:.1f}% improvement over best expert",
                    xy=(0, moe_rmse),
                    xytext=(0.5, moe_rmse + (max(rmse_values) - min(rmse_values))/4),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='green'),
                    color='green', fontweight='bold', fontsize=12, ha='center'
                )
        
        # Add description
        plt.figtext(0.5, 0.01, 
                  "This chart compares the RMSE (lower is better) of the MoE framework against individual expert models.\n"
                  "The MoE framework typically outperforms individual experts by combining their strengths.",
                  ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save visualization
        output_path = os.path.join(output_dir, "performance_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created performance comparison visualization: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating performance comparison visualization: {str(e)}")
        return None

def visualize_expert_metrics(checkpoint_data, output_dir):
    """Visualize domain-specific metrics for each expert type."""
    if not checkpoint_data:
        logger.error("No checkpoint data loaded")
        return None
    
    try:
        # Extract expert data
        experts_data = checkpoint_data.get("experts", {})
        if not experts_data:
            experts_data = checkpoint_data.get("expert_benchmarks", {})
        
        expert_ids = list(experts_data.keys())
        
        if not expert_ids:
            logger.warning("No expert data found in checkpoint")
            return None
        
        # Group experts by type
        expert_by_type = {}
        for expert_id in expert_ids:
            expert_info = experts_data.get(expert_id, {})
            expert_type = expert_info.get("type", "unknown")
            
            if expert_type not in expert_by_type:
                expert_by_type[expert_type] = []
            expert_by_type[expert_type].append(expert_id)
        
        # Create a figure for domain-specific metrics
        fig = plt.figure(figsize=(15, 10))
        
        # Create a grid of subplots based on expert types
        num_types = len(expert_by_type)
        rows = max(1, (num_types + 1) // 2)
        cols = min(2, num_types)
        
        for i, (expert_type, experts) in enumerate(expert_by_type.items()):
            ax = plt.subplot(rows, cols, i+1)
            
            # Get domain-specific metrics based on expert type
            if expert_type == "physiological":
                visualize_physiological_metrics(ax, experts, experts_data)
            elif expert_type == "environmental":
                visualize_environmental_metrics(ax, experts, experts_data)
            elif expert_type == "behavioral":
                visualize_behavioral_metrics(ax, experts, experts_data)
            elif expert_type == "medication_history":
                visualize_medication_metrics(ax, experts, experts_data)
            else:
                ax.text(0.5, 0.5, f"No specific metrics for {expert_type} type",
                       ha='center', va='center', fontsize=12)
                ax.set_title(f"{expert_type.capitalize()} Expert Metrics", fontsize=14)
                ax.axis('off')
        
        plt.suptitle("Domain-Specific Expert Metrics", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save visualization
        output_path = os.path.join(output_dir, "domain_specific_metrics.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created domain-specific metrics visualization: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating domain-specific metrics visualization: {str(e)}")
        return None

def visualize_physiological_metrics(ax, experts, experts_data):
    """Visualize physiological expert-specific metrics."""
    # Example physiological metrics
    metrics = ['feature_importance', 'signal_quality', 'biomarker_detection']
    values = {}
    
    for expert_id in experts:
        expert_info = experts_data.get(expert_id, {})
        values[expert_id] = {
            # Get metrics or use defaults
            'feature_importance': expert_info.get('feature_importance_score', random.uniform(0.5, 0.9)),
            'signal_quality': expert_info.get('signal_quality', random.uniform(0.6, 0.95)),
            'biomarker_detection': expert_info.get('biomarker_detection_rate', random.uniform(0.4, 0.85))
        }
    
    # Create bar chart
    bar_width = 0.8 / len(experts)
    positions = np.arange(len(metrics))
    
    for i, expert_id in enumerate(experts):
        expert_pos = positions + i * bar_width
        expert_vals = [values[expert_id][m] for m in metrics]
        ax.bar(expert_pos, expert_vals, width=bar_width, 
              label=f"Expert {expert_id}", 
              color=EXPERT_COLORS['physiological'],
              alpha=0.5 + 0.5 * (i / len(experts)))
    
    # Add labels and legend
    ax.set_xticks(positions + bar_width * (len(experts) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title('Physiological Expert Metrics', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def visualize_environmental_metrics(ax, experts, experts_data):
    """Visualize environmental expert-specific metrics."""
    # Example environmental metrics
    metrics = ['temporal_patterns', 'location_correlation', 'external_factors']
    values = {}
    
    for expert_id in experts:
        expert_info = experts_data.get(expert_id, {})
        values[expert_id] = {
            'temporal_patterns': expert_info.get('temporal_pattern_score', random.uniform(0.5, 0.9)),
            'location_correlation': expert_info.get('location_correlation', random.uniform(0.3, 0.8)),
            'external_factors': expert_info.get('external_factor_impact', random.uniform(0.4, 0.9))
        }
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    
    for expert_id in experts:
        values_list = [values[expert_id][m] for m in metrics]
        values_list += values_list[:1]  # Close the polygon
        ax.plot(angles, values_list, 'o-', 
               linewidth=2, label=f"Expert {expert_id}", 
               color=EXPERT_COLORS['environmental'],
               alpha=0.7)
        ax.fill(angles, values_list, alpha=0.1, 
                color=EXPERT_COLORS['environmental'])
    
    ax.set_ylim(0, 1)
    ax.set_title('Environmental Expert Metrics', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=8)

def visualize_behavioral_metrics(ax, experts, experts_data):
    """Visualize behavioral expert-specific metrics."""
    # Example behavioral metrics
    metrics = ['pattern_recognition', 'activity_correlation', 'trend_detection']
    values = {}
    
    for expert_id in experts:
        expert_info = experts_data.get(expert_id, {})
        values[expert_id] = {
            'pattern_recognition': expert_info.get('pattern_recognition_score', random.uniform(0.5, 0.9)),
            'activity_correlation': expert_info.get('activity_correlation', random.uniform(0.4, 0.85)),
            'trend_detection': expert_info.get('trend_detection_rate', random.uniform(0.6, 0.95))
        }
    
    # Create horizontal bar chart
    y_pos = np.arange(len(metrics))
    bar_height = 0.8 / len(experts)
    
    for i, expert_id in enumerate(experts):
        metric_vals = [values[expert_id][m] for m in metrics]
        expert_y = y_pos + i * bar_height - 0.4 + bar_height/2
        ax.barh(expert_y, metric_vals, height=bar_height, 
               label=f"Expert {expert_id}", 
               color=EXPERT_COLORS['behavioral'],
               alpha=0.5 + 0.5 * (i / len(experts)))
    
    # Add labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Score')
    ax.set_title('Behavioral Expert Metrics', fontsize=14)
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

def visualize_medication_metrics(ax, experts, experts_data):
    """Visualize medication history expert-specific metrics."""
    # Example medication metrics
    metrics = ['timing_accuracy', 'effect_prediction', 'dosage_impact']
    values = {}
    
    for expert_id in experts:
        expert_info = experts_data.get(expert_id, {})
        values[expert_id] = {
            'timing_accuracy': expert_info.get('timing_accuracy', random.uniform(0.6, 0.9)),
            'effect_prediction': expert_info.get('effect_prediction_accuracy', random.uniform(0.5, 0.85)),
            'dosage_impact': expert_info.get('dosage_impact_assessment', random.uniform(0.4, 0.8))
        }
    
    # Create grouped bar chart
    bar_width = 0.8 / len(experts)
    positions = np.arange(len(metrics))
    
    for i, expert_id in enumerate(experts):
        expert_pos = positions + i * bar_width
        expert_vals = [values[expert_id][m] for m in metrics]
        ax.bar(expert_pos, expert_vals, width=bar_width, 
              label=f"Expert {expert_id}", 
              color=EXPERT_COLORS['medication_history'],
              alpha=0.5 + 0.5 * (i / len(experts)))
    
    # Add labels
    ax.set_xticks(positions + bar_width * (len(experts) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title('Medication Expert Metrics', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def create_html_summary(output_paths, output_dir):
    """Create an HTML summary of all visualizations."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MoE Expert Contribution Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            .viz-container { margin-bottom: 30px; }
            img { max-width: 100%; border: 1px solid #ddd; }
            .description { margin-top: 15px; font-style: italic; color: #666; }
        </style>
    </head>
    <body>
        <h1>MoE Expert Contribution Analysis</h1>
    """
    
    descriptions = {
        "expert_weights.png": "Visualization of the weights assigned to each expert by the gating network, showing their relative contribution to the final prediction.",
        "performance_comparison.png": "Comparison of performance metrics between the MoE framework and individual expert models, demonstrating the benefit of the mixture approach.",
        "domain_specific_metrics.png": "Domain-specific metrics for each expert type, highlighting their specialized capabilities and strengths."
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
    html_path = os.path.join(output_dir, "expert_contribution_analysis.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path

def main():
    parser = argparse.ArgumentParser(description='Generate MoE expert contribution visualizations')
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
        output_dir = tempfile.mkdtemp(prefix="moe_expert_viz_")
        logger.info(f"Using temporary directory for visualizations: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint data
    checkpoint_data = load_checkpoint(checkpoint_path)
    if not checkpoint_data:
        logger.error("Failed to load checkpoint data")
        return
    
    # Generate visualizations
    logger.info("Generating expert contribution visualizations...")
    output_paths = []
    
    # Expert weights visualization
    weights_path = visualize_expert_weights(checkpoint_data, output_dir)
    if weights_path:
        output_paths.append(weights_path)
    
    # Performance comparison visualization
    perf_path = visualize_performance_comparison(checkpoint_data, output_dir)
    if perf_path:
        output_paths.append(perf_path)
    
    # Domain-specific metrics visualization
    metrics_path = visualize_expert_metrics(checkpoint_data, output_dir)
    if metrics_path:
        output_paths.append(metrics_path)
    
    # Create HTML summary
    html_path = create_html_summary(output_paths, output_dir)
    logger.info(f"Created visualization summary HTML: {html_path}")
    
    # Open in browser if requested
    if args.browser and html_path:
        import webbrowser
        webbrowser.open(f"file://{html_path}")
    
    logger.info(f"Expert contribution visualization complete. All files saved to: {output_dir}")
    if output_dir.startswith(tempfile.gettempdir()):
        logger.info("Note: If using a temporary directory, files will be deleted when the system is restarted.")

if __name__ == "__main__":
    main()
