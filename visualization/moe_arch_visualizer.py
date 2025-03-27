"""
MoE Architecture Visualization Module

This module provides visualizations that accurately reflect the MoE framework architecture:
1. Expert Models (domain-specific experts)
2. Gating Network (expert selection mechanism)
3. Integration Flow (how components connect)
4. End-to-End Performance (overall system metrics)

All visualizations follow the actual implementation patterns in the framework.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import logging
import networkx as nx
from datetime import datetime
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Configure logging
logger = logging.getLogger(__name__)

class MoEArchVisualizer:
    """Architecture-focused visualizer for MoE framework results."""
    
    def __init__(self, checkpoint_path=None, use_temp_dir=True, output_dir=None):
        """
        Initialize the MoE architecture visualizer.
        
        Args:
            checkpoint_path: Path to checkpoint JSON file
            use_temp_dir: Whether to use a temporary directory for outputs
            output_dir: Custom output directory (ignored if use_temp_dir is True)
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_data = None
        self.use_temp_dir = use_temp_dir
        
        # Set up output directory
        if use_temp_dir:
            self.output_dir = tempfile.mkdtemp(prefix="moe_arch_viz_")
            logger.info(f"Using temporary directory for visualizations: {self.output_dir}")
        else:
            self.output_dir = output_dir or "./visualization_output"
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Using custom directory for visualizations: {self.output_dir}")
        
        # Track generated visualizations
        self.visualizations = {
            "expert_models": [],
            "gating_network": [],
            "integration_flow": [],
            "performance": []
        }
        
        # Set default style
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        
        # Color schemes for different expert types
        self.expert_colors = {
            "physiological": "#2C7BB6",  # Blue
            "environmental": "#7FBC41",  # Green
            "behavioral": "#D7301F",     # Red
            "medication_history": "#B15928"  # Brown
        }
        
        # Load checkpoint data if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint data from JSON file.
        
        Args:
            checkpoint_path: Path to checkpoint JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(checkpoint_path, 'r') as f:
                self.checkpoint_data = json.load(f)
            logger.info(f"Loaded checkpoint data from {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading checkpoint data: {str(e)}")
            return False
    
    def visualize_all(self):
        """Generate all architecture-focused visualizations."""
        # Expert model visualizations
        self.visualize_expert_structure()
        self.visualize_expert_performance()
        self.visualize_domain_specific_metrics()
        
        # Gating network visualizations
        self.visualize_gating_mechanism()
        self.visualize_expert_selection()
        self.visualize_weight_distribution()
        
        # Integration flow visualizations
        self.visualize_architecture_diagram()
        self.visualize_data_flow()
        
        # End-to-end performance visualizations
        self.visualize_overall_performance()
        self.visualize_component_contribution()
        
        # Create comprehensive summary
        self.create_visualization_summary()
        
        return self.visualizations
    
    def visualize_expert_structure(self):
        """Visualize the expert model structure in the MoE framework."""
        if not self.checkpoint_data:
            logger.error("No checkpoint data loaded")
            return None
        
        # Extract expert model information
        try:
            experts_data = self.checkpoint_data.get("experts", {})
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
            colors = [self.expert_colors.get(t, "#CCCCCC") for t in types]
            
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
            output_path = os.path.join(self.output_dir, "expert_structure.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Created expert structure visualization: {output_path}")
            self.visualizations["expert_models"].append({
                "title": "Expert Model Structure",
                "path": output_path,
                "description": "Visualization of expert model domains in the MoE framework"
            })
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating expert structure visualization: {str(e)}")
            return None
    
    def visualize_expert_performance(self):
        """Visualize performance metrics for each expert in the MoE framework."""
        if not self.checkpoint_data:
            logger.error("No checkpoint data loaded")
            return None
        
        try:
            experts_data = self.checkpoint_data.get("experts", {})
            expert_ids = list(experts_data.keys())
            
            if not expert_ids:
                logger.warning("No expert data found in checkpoint")
                return None
            
            # Extract expert performance metrics
            metrics = ["rmse", "mae", "r2"]
            expert_metrics = {}
            
            for expert_id in expert_ids:
                expert_info = experts_data.get(expert_id, {})
                expert_metrics[expert_id] = {
                    "rmse": float(expert_info.get("rmse", 0)),
                    "mae": float(expert_info.get("mae", 0)),
                    "r2": float(expert_info.get("r2", 0)),
                    "type": expert_info.get("type", "unknown")
                }
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            
            # For each metric, create a bar chart
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                # Extract values and sort by expert type
                expert_types = [expert_metrics[e]["type"] for e in expert_ids]
                values = [expert_metrics[e][metric] for e in expert_ids]
                colors = [self.expert_colors.get(expert_metrics[e]["type"], "#CCCCCC") for e in expert_ids]
                
                # Create bar chart
                bars = ax.bar(expert_ids, values, color=colors)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f"{height:.3f}", ha='center', va='bottom', fontsize=9)
                
                # Set titles and labels
                metric_names = {"rmse": "RMSE", "mae": "MAE", "r2": "R²"}
                ax.set_title(f"Expert {metric_names.get(metric, metric)}", fontsize=14)
                ax.set_xlabel("Expert ID", fontsize=12)
                ax.set_ylabel(metric_names.get(metric, metric), fontsize=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Add grid for readability
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add color legend for expert types
                if i == 2:  # Only add legend to the last plot
                    legend_patches = [mpatches.Patch(color=self.expert_colors.get(t, "#CCCCCC"), 
                                                  label=f"{t.capitalize()} Expert") 
                                   for t in set(expert_types)]
                    ax.legend(handles=legend_patches, loc="upper right")
            
            plt.suptitle("Expert Model Performance Metrics", fontsize=16, y=1.05)
            plt.tight_layout()
            
            # Save visualization
            output_path = os.path.join(self.output_dir, "expert_performance.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Created expert performance visualization: {output_path}")
            self.visualizations["expert_models"].append({
                "title": "Expert Performance Metrics",
                "path": output_path,
                "description": "Comparison of performance metrics across different experts"
            })
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating expert performance visualization: {str(e)}")
            return None
