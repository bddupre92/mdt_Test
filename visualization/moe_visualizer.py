"""
MoE Visualization Module

This module provides architecture-focused visualization capabilities for the MoE pipeline,
including:
1. Expert Models Visualization - Performance and domain-specific metrics for each expert
2. Gating Network Analysis - How experts are selected and weighted
3. Integration Flow - Architecture diagram and data flow through the system
4. Performance Analysis - Overall metrics and expert contributions

All visualizations are saved to temporary directories and organized in an HTML summary.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import logging
import random
from datetime import datetime
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from matplotlib.path import Path as MatplotlibPath
import matplotlib.patches as patches

# Configure logging
logger = logging.getLogger(__name__)

class MoEVisualizer:
    """Architecture-focused visualizer for MoE pipeline results based on checkpoint data.
    
    This visualizer creates diagrams and charts that reflect the actual architecture of the
    MoE framework, with its expert models, gating network, and integration components.
    """
    
    def __init__(self, checkpoint_path=None, use_temp_dir=True, output_dir=None):
        """
        Initialize the MoE visualizer.
        
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
            self.output_dir = tempfile.mkdtemp(prefix="moe_viz_")
            logger.info(f"Using temporary directory for visualizations: {self.output_dir}")
        else:
            self.output_dir = output_dir or "./visualization_output"
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Using custom directory for visualizations: {self.output_dir}")
        
        # Track generated visualizations
        self.visualizations = {
            "expert_models": [],
            "gating_network": [],
            "architecture_flow": [],
            "data_flow": [],
            "performance": []
        }
        
        # Component colors for architecture visualization
        self.component_colors = {
            "data": "#8073AC",           # Purple
            "data_quality": "#FDB863",    # Orange
            "expert": "#B2DF8A",          # Light green
            "gating": "#A6CEE3",          # Light blue
            "integration": "#FB9A99",     # Light red
            "meta_learner": "#FDBF6F",    # Light orange
            "output": "#CAB2D6"           # Light purple
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
    
    def create_expert_benchmarks(self):
        """
        Create visualizations for expert model benchmarks.
        
        Returns:
            List of paths to generated visualizations
        """
        if not self.checkpoint_data:
            logger.error("No checkpoint data loaded")
            return []
        
        viz_paths = []
        
        try:
            # Extract expert benchmark data
            expert_data = self.checkpoint_data.get('expert_benchmarks', {})
            expert_ids = list(expert_data.keys())
            
            if not expert_ids:
                logger.warning("No expert benchmark data found in checkpoint")
                return []
            
            # 1. Performance comparison bar chart
            metrics = ['rmse', 'mae', 'r2']
            metric_values = {}
            
            for metric in metrics:
                metric_values[metric] = [expert_data[expert_id].get('metrics', {}).get(metric, 0) 
                                        for expert_id in expert_ids]
            
            # Create a figure with subplots for each metric
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for i, metric in enumerate(metrics):
                axes[i].bar(expert_ids, metric_values[metric], color=sns.color_palette("husl", len(expert_ids)))
                axes[i].set_title(f"Expert {metric.upper()} Comparison")
                axes[i].set_ylabel(metric.upper())
                axes[i].set_xlabel("Expert Model")
                if metric == 'r2':  # For R², higher is better
                    for j, v in enumerate(metric_values[metric]):
                        axes[i].text(j, v + 0.02, f"{v:.2f}", ha='center')
                else:  # For RMSE and MAE, lower is better
                    for j, v in enumerate(metric_values[metric]):
                        axes[i].text(j, v + 0.02, f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.output_dir, "expert_performance_comparison.png")
            plt.savefig(fig_path)
            plt.close(fig)
            
            viz_paths.append(fig_path)
            logger.info(f"Created expert performance comparison chart: {fig_path}")
            
            # 2. Radar chart for expert metrics
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Set angles for each metric (divide the plot by number of variables)
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Close the loop
            
            # Plot each expert
            for i, expert_id in enumerate(expert_ids):
                values = [expert_data[expert_id].get('metrics', {}).get(metric, 0) for metric in metrics]
                # Normalize values between 0 and 1 for radar chart
                if metric == 'r2':  # Higher is better for R²
                    values = [v / max([expert_data[e].get('metrics', {}).get(metric, 0.1) for e in expert_ids]) for v, metric in zip(values, metrics)]
                else:  # Lower is better for RMSE and MAE
                    max_vals = [max([expert_data[e].get('metrics', {}).get(metric, 0.1) for e in expert_ids]) for metric in metrics]
                    values = [1 - (v / max_val) if max_val > 0 else 0 for v, max_val in zip(values, max_vals)]
                
                values += values[:1]  # Close the loop
                
                # Draw the expert line
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=expert_id)
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels and title
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_title("Expert Model Metrics Comparison")
            plt.legend(loc='upper right')
            
            # Save radar chart
            radar_path = os.path.join(self.output_dir, "expert_metrics_radar.png")
            plt.savefig(radar_path)
            plt.close(fig)
            
            viz_paths.append(radar_path)
            logger.info(f"Created expert metrics radar chart: {radar_path}")
            
            # 3. Prediction vs. Actual Visualization
            if 'predictions' in self.checkpoint_data:
                predictions = self.checkpoint_data['predictions']
                actuals = self.checkpoint_data.get('actuals', [])
                
                if predictions and actuals and len(predictions) == len(actuals):
                    plt.figure(figsize=(12, 8))
                    
                    # Create scatter plot with prediction vs actual values
                    plt.scatter(actuals, predictions, alpha=0.7)
                    
                    # Plot the ideal line (y=x)
                    min_val = min(min(predictions), min(actuals))
                    max_val = max(max(predictions), max(actuals))
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                    
                    plt.title('Prediction vs. Actual Values')
                    plt.xlabel('Actual Values')
                    plt.ylabel('Predicted Values')
                    plt.grid(True)
                    
                    # Add R² value
                    from sklearn.metrics import r2_score
                    r2 = r2_score(actuals, predictions)
                    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                             bbox=dict(facecolor='white', alpha=0.8))
                    
                    pred_actual_path = os.path.join(self.output_dir, "prediction_vs_actual.png")
                    plt.savefig(pred_actual_path)
                    plt.close()
                    
                    viz_paths.append(pred_actual_path)
                    logger.info(f"Created prediction vs. actual visualization: {pred_actual_path}")
                    
                    # Create individual expert prediction plots
                    if 'expert_predictions' in self.checkpoint_data:
                        expert_predictions = self.checkpoint_data['expert_predictions']
                        
                        plt.figure(figsize=(14, 10))
                        
                        # Plot actuals as a line
                        plt.plot(range(len(actuals)), actuals, 'k-', linewidth=2, label='Actual')
                        
                        # Plot each expert's predictions
                        for expert_id in expert_ids:
                            if expert_id in expert_predictions and len(expert_predictions[expert_id]) == len(actuals):
                                plt.plot(range(len(actuals)), expert_predictions[expert_id], '--', linewidth=1.5, label=expert_id)
                        
                        plt.title('Expert Predictions vs. Actual Values')
                        plt.xlabel('Data Point Index')
                        plt.ylabel('Value')
                        plt.legend()
                        plt.grid(True)
                        
                        expert_pred_path = os.path.join(self.output_dir, "expert_predictions_comparison.png")
                        plt.savefig(expert_pred_path)
                        plt.close()
                        
                        viz_paths.append(expert_pred_path)
                        logger.info(f"Created expert predictions comparison: {expert_pred_path}")
            
            # 4. Expert Performance Over Time
            if 'performance_over_time' in self.checkpoint_data:
                perf_time_data = self.checkpoint_data['performance_over_time']
                
                if isinstance(perf_time_data, dict) and 'timestamps' in perf_time_data and 'metrics' in perf_time_data:
                    timestamps = perf_time_data['timestamps']
                    metrics_by_expert = perf_time_data['metrics']
                    
                    # Plot RMSE over time for each expert
                    plt.figure(figsize=(14, 8))
                    
                    for expert_id in expert_ids:
                        if expert_id in metrics_by_expert and 'rmse' in metrics_by_expert[expert_id]:
                            rmse_values = metrics_by_expert[expert_id]['rmse']
                            if len(rmse_values) == len(timestamps):
                                plt.plot(timestamps, rmse_values, marker='o', linestyle='-', label=expert_id)
                    
                    plt.title('Expert RMSE Over Time')
                    plt.xlabel('Time')
                    plt.ylabel('RMSE (lower is better)')
                    plt.legend()
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    
                    expert_time_path = os.path.join(self.output_dir, "expert_performance_over_time.png")
                    plt.savefig(expert_time_path, bbox_inches='tight')
                    plt.close()
                    
                    viz_paths.append(expert_time_path)
                    logger.info(f"Created expert performance over time chart: {expert_time_path}")
            
            # 5. Feature importance heatmap if available
            features_data = {}
            for expert_id in expert_ids:
                if 'feature_importances' in expert_data[expert_id]:
                    features_data[expert_id] = expert_data[expert_id]['feature_importances']
            
            if features_data:
                # Convert to DataFrame for heatmap
                all_features = set()
                for expert_id in features_data:
                    all_features.update(features_data[expert_id].keys())
                
                # Create DataFrame with feature importances per expert
                importance_df = pd.DataFrame(0, index=list(all_features), columns=list(features_data.keys()))
                
                for expert_id in features_data:
                    for feature, importance in features_data[expert_id].items():
                        importance_df.loc[feature, expert_id] = importance
                
                # Create heatmap
                plt.figure(figsize=(12, max(8, len(all_features) * 0.3)))
                sns.heatmap(importance_df, cmap="YlGnBu", annot=False)
                plt.title("Feature Importance by Expert Model")
                plt.tight_layout()
                
                # Save heatmap
                heatmap_path = os.path.join(self.output_dir, "feature_importance_heatmap.png")
                plt.savefig(heatmap_path)
                plt.close()
                
                viz_paths.append(heatmap_path)
                logger.info(f"Created feature importance heatmap: {heatmap_path}")
                
                # 6. Top features barplot for each expert
                for expert_id in features_data:
                    feature_importance = features_data[expert_id]
                    # Sort features by importance and get top N
                    top_n = 10  # Number of top features to show
                    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n])
                    
                    plt.figure(figsize=(10, 6))
                    plt.barh(list(sorted_features.keys()), list(sorted_features.values()))
                    plt.xlabel('Importance')
                    plt.title(f'Top {top_n} Features for {expert_id}')
                    plt.tight_layout()
                    
                    top_features_path = os.path.join(self.output_dir, f"{expert_id}_top_features.png")
                    plt.savefig(top_features_path)
                    plt.close()
                    
                    viz_paths.append(top_features_path)
                    logger.info(f"Created top features visualization for {expert_id}: {top_features_path}")
            
            # 7. Expert Analysis Summary
            # Create a table-like figure with expert recommendations
            expert_recommendations = {}
            for expert_id in expert_ids:
                metrics = expert_data[expert_id].get('metrics', {})
                strengths = []
                weaknesses = []
                
                # Analyze RMSE
                rmse = metrics.get('rmse', 0)
                avg_rmse = sum([expert_data[e].get('metrics', {}).get('rmse', 0) for e in expert_ids]) / len(expert_ids) if expert_ids else 0
                if rmse < avg_rmse:
                    strengths.append("Lower than average error")
                else:
                    weaknesses.append("Higher than average error")
                
                # Analyze R²
                r2 = metrics.get('r2', 0)
                if r2 > 0.7:
                    strengths.append("Strong predictive power")
                elif r2 < 0.5:
                    weaknesses.append("Limited predictive power")
                
                # Check training time if available
                train_time = expert_data[expert_id].get('training_time', 0)
                if train_time:
                    # Convert all training times to float to avoid type issues
                    try:
                        train_time_float = float(train_time)
                        all_times = []
                        for e in expert_ids:
                            e_time = expert_data[e].get('training_time', 0)
                            try:
                                all_times.append(float(e_time))
                            except (ValueError, TypeError):
                                all_times.append(0.0)
                        
                        avg_time = sum(all_times) / len(all_times) if all_times else 0
                        if train_time_float < avg_time:
                            strengths.append("Fast training time")
                        else:
                            weaknesses.append("Slow training time")
                    except (ValueError, TypeError):
                        # If conversion fails, skip this analysis
                        pass
                
                # Generate recommendations
                recommendations = []
                if len(weaknesses) > len(strengths):
                    recommendations.append("Consider model optimization")
                    recommendations.append("May benefit from hyperparameter tuning")
                else:
                    recommendations.append("Performs well in current configuration")
                
                # Store results
                expert_recommendations[expert_id] = {
                    'strengths': strengths,
                    'weaknesses': weaknesses,
                    'recommendations': recommendations
                }
            
            # Create the summary table visualization
            summary_rows = len(expert_ids) + 1  # +1 for header
            summary_cols = 4  # Expert, Strengths, Weaknesses, Recommendations
            
            fig, ax = plt.subplots(figsize=(16, 2 * summary_rows))
            ax.axis('tight')
            ax.axis('off')
            
            # Create the table data
            table_data = [['Expert', 'Strengths', 'Weaknesses', 'Recommendations']]
            for expert_id in expert_ids:
                rec = expert_recommendations[expert_id]
                strengths_text = '\n'.join(rec['strengths']) if rec['strengths'] else 'None identified'
                weaknesses_text = '\n'.join(rec['weaknesses']) if rec['weaknesses'] else 'None identified'
                recommendations_text = '\n'.join(rec['recommendations'])
                table_data.append([expert_id, strengths_text, weaknesses_text, recommendations_text])
            
            # Create the table
            table = ax.table(cellText=table_data, colWidths=[0.2, 0.3, 0.3, 0.3], loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Style the header row - using direct indices to avoid slice indexing issues
            for j in range(len(table_data[0])):
                cell = table[(0, j)]
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold')
            
            # Style alternating rows
            for i in range(1, len(table_data)):
                for j in range(len(table_data[0])):
                    try:
                        if i % 2 == 0:
                            table[(i, j)].set_facecolor('#D9E1F2')
                    except KeyError:
                        # Skip if the cell doesn't exist
                        pass
            
            plt.title('Expert Model Analysis Summary', fontsize=14, pad=20)
            plt.tight_layout()
            
            summary_path = os.path.join(self.output_dir, "expert_analysis_summary.png")
            plt.savefig(summary_path, bbox_inches='tight')
            plt.close()
            
            viz_paths.append(summary_path)
            logger.info(f"Created expert analysis summary: {summary_path}")
            
            return viz_paths
        
        except Exception as e:
            logger.error(f"Error creating expert benchmarks: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return viz_paths
    
    def create_gating_visualizations(self):
        """
        Create visualizations for gating network analysis.
        
        Returns:
            List of paths to generated visualizations
        """
        if not self.checkpoint_data:
            logger.error("No checkpoint data loaded")
            return []
        
        viz_paths = []
        
        try:
            # Extract gating evaluation data
            gating_data = self.checkpoint_data.get('gating_evaluation', {})
            
            if not gating_data:
                logger.warning("No gating network data found in checkpoint")
                return []
            
            # 1. Expert selection frequency pie chart
            selection_frequencies = gating_data.get('expert_selection_frequencies', {})
            
            if selection_frequencies:
                plt.figure(figsize=(10, 8))
                plt.pie(selection_frequencies.values(), labels=selection_frequencies.keys(), 
                        autopct='%1.1f%%', startangle=90, shadow=True)
                plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
                plt.title('Expert Selection Frequency')
                
                # Save pie chart
                pie_path = os.path.join(self.output_dir, "expert_selection_frequency.png")
                plt.savefig(pie_path)
                plt.close()
                
                viz_paths.append(pie_path)
                logger.info(f"Created expert selection frequency pie chart: {pie_path}")
                
                # Also create a bar chart version
                plt.figure(figsize=(12, 6))
                experts = list(selection_frequencies.keys())
                frequencies = list(selection_frequencies.values())
                bars = plt.bar(experts, frequencies, color=sns.color_palette("viridis", len(experts)))
                
                plt.title('Expert Selection Frequency')
                plt.xlabel('Expert Model')
                plt.ylabel('Selection Frequency (%)')
                
                # Add percentage labels
                total = sum(frequencies)
                for bar, freq in zip(bars, frequencies):
                    height = bar.get_height()
                    percentage = (freq / total) * 100 if total > 0 else 0
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{percentage:.1f}%', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Save the bar chart
                bar_path = os.path.join(self.output_dir, "expert_selection_frequency_bar.png")
                plt.savefig(bar_path)
                plt.close()
                
                viz_paths.append(bar_path)
                logger.info(f"Created expert selection frequency bar chart: {bar_path}")
            
            # 2. Decision regret analysis (with more detail)
            regret_data = gating_data.get('regret_analysis', {})
            
            if regret_data:
                # Mean regret by expert
                experts = list(regret_data.keys())
                mean_regret = [regret_data[expert].get('mean_regret', 0) for expert in experts]
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(experts, mean_regret, color=sns.color_palette("rocket", len(experts)))
                
                plt.title('Mean Decision Regret by Expert')
                plt.xlabel('Expert Model')
                plt.ylabel('Mean Regret (lower is better)')
                
                # Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Save mean regret chart
                mean_regret_path = os.path.join(self.output_dir, "mean_decision_regret.png")
                plt.savefig(mean_regret_path)
                plt.close()
                
                viz_paths.append(mean_regret_path)
                logger.info(f"Created mean decision regret chart: {mean_regret_path}")
                
                # Regret distribution if available
                if any('regret_distribution' in regret_data.get(expert, {}) for expert in experts):
                    plt.figure(figsize=(14, 8))
                    
                    # Plot regret distributions as boxplots
                    box_data = []
                    box_labels = []
                    
                    for expert in experts:
                        if 'regret_distribution' in regret_data[expert]:
                            box_data.append(regret_data[expert]['regret_distribution'])
                            box_labels.append(expert)
                    
                    if box_data:
                        plt.boxplot(box_data, labels=box_labels, patch_artist=True)
                        plt.title('Decision Regret Distribution by Expert')
                        plt.xlabel('Expert Model')
                        plt.ylabel('Regret Distribution')
                        plt.grid(True, alpha=0.3)
                        
                        # Save regret distribution chart
                        regret_dist_path = os.path.join(self.output_dir, "regret_distribution.png")
                        plt.savefig(regret_dist_path)
                        plt.close()
                        
                        viz_paths.append(regret_dist_path)
                        logger.info(f"Created regret distribution chart: {regret_dist_path}")
                
                # Optimal vs Selected expert visualization if available
                if 'optimal_selections' in gating_data and 'actual_selections' in gating_data:
                    optimal = gating_data['optimal_selections']
                    actual = gating_data['actual_selections']
                    
                    if len(optimal) == len(actual):
                        # Count matches and mismatches
                        matches = sum(1 for o, a in zip(optimal, actual) if o == a)
                        mismatches = len(optimal) - matches
                        match_rate = matches / len(optimal) if len(optimal) > 0 else 0
                        
                        # Create a plot showing optimal selections vs actual selections
                        plt.figure(figsize=(14, 7))
                        
                        # Create a 2D array for the heatmap
                        all_experts = sorted(list(set(optimal + actual)))
                        confusion = np.zeros((len(all_experts), len(all_experts)))
                        
                        for o, a in zip(optimal, actual):
                            o_idx = all_experts.index(o)
                            a_idx = all_experts.index(a)
                            confusion[o_idx, a_idx] += 1
                            
                        # Plot heatmap
                        sns.heatmap(confusion, annot=True, fmt='.0f', xticklabels=all_experts,
                                   yticklabels=all_experts, cmap="YlGnBu")
                        plt.title(f'Optimal vs. Selected Expert\nMatch Rate: {match_rate:.2%}')
                        plt.xlabel('Selected Expert')
                        plt.ylabel('Optimal Expert')
                        plt.tight_layout()
                        
                        # Save optimal vs selected chart
                        optimal_vs_selected_path = os.path.join(self.output_dir, "optimal_vs_selected_experts.png")
                        plt.savefig(optimal_vs_selected_path)
                        plt.close()
                        
                        viz_paths.append(optimal_vs_selected_path)
                        logger.info(f"Created optimal vs selected experts chart: {optimal_vs_selected_path}")
            
            # 3. Expert routing patterns (Sankey diagram)
            # This needs networkx and matplotlib for Sankey
            routing_data = gating_data.get('routing_patterns', {})
            
            if routing_data and 'transitions' in routing_data:
                try:
                    import networkx as nx
                    from matplotlib.sankey import Sankey
                    
                    transitions = routing_data['transitions']
                    
                    # Convert transitions to a graph
                    G = nx.DiGraph()
                    
                    # Add nodes and edges
                    for source, targets in transitions.items():
                        for target, count in targets.items():
                            if count > 0:  # Only add edges with positive count
                                G.add_edge(source, target, weight=count)
                    
                    # Draw the graph
                    plt.figure(figsize=(14, 10))
                    pos = nx.spring_layout(G, seed=42)
                    
                    # Draw nodes
                    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
                    
                    # Draw edges with width proportional to weight
                    weights = [G[u][v]['weight'] / 10 for u, v in G.edges()]
                    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', 
                                          arrowsize=20, connectionstyle='arc3,rad=0.1')
                    
                    # Draw labels
                    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
                    
                    plt.title('Expert Routing Patterns')
                    plt.axis('off')
                    
                    # Save routing patterns chart
                    routing_path = os.path.join(self.output_dir, "expert_routing_patterns.png")
                    plt.savefig(routing_path, bbox_inches='tight')
                    plt.close()
                    
                    viz_paths.append(routing_path)
                    logger.info(f"Created expert routing patterns chart: {routing_path}")
                except ImportError:
                    logger.warning("Could not create routing patterns visualization; networkx is required")
            
            # 4. Weight distribution over time
            weights_over_time = gating_data.get('weights_over_time', {})
            
            if weights_over_time and 'timestamps' in weights_over_time and 'weights' in weights_over_time:
                timestamps = weights_over_time['timestamps']
                weights = weights_over_time['weights']
                
                # Convert to DataFrame for easier plotting
                experts = list(weights[0].keys()) if weights and isinstance(weights[0], dict) else []
                
                if experts:
                    # Create DataFrame from the weights data
                    weight_data = []
                    for i, ts in enumerate(timestamps):
                        for expert in experts:
                            weight_data.append({
                                'timestamp': ts,
                                'expert': expert,
                                'weight': weights[i].get(expert, 0)
                            })
                    
                    weight_df = pd.DataFrame(weight_data)
                    
                    # Create stacked area chart
                    plt.figure(figsize=(14, 7))
                    
                    # Convert timestamps to datetime if they are strings
                    if isinstance(weight_df['timestamp'][0], str):
                        try:
                            weight_df['timestamp'] = pd.to_datetime(weight_df['timestamp'])
                        except:
                            # If conversion fails, use numeric indices instead
                            weight_df['timestamp'] = range(len(timestamps))
                    
                    # Pivot the data for area chart
                    pivot_df = weight_df.pivot(index='timestamp', columns='expert', values='weight')
                    
                    # Plot stacked area chart
                    pivot_df.plot.area(stacked=True, alpha=0.7, ax=plt.gca())
                    
                    plt.title('Expert Weight Distribution Over Time')
                    plt.xlabel('Time')
                    plt.ylabel('Weight')
                    plt.legend(title='Experts', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Save weights chart
                    weights_path = os.path.join(self.output_dir, "expert_weights_over_time.png")
                    plt.savefig(weights_path)
                    plt.close()
                    
                    viz_paths.append(weights_path)
                    logger.info(f"Created expert weights over time chart: {weights_path}")
                    
                    # Also create line charts for each expert's weight
                    plt.figure(figsize=(14, 7))
                    
                    for expert in experts:
                        expert_weights = [w.get(expert, 0) for w in weights]
                        plt.plot(timestamps, expert_weights, marker='o', linestyle='-', label=expert)
                    
                    plt.title('Expert Weights Over Time')
                    plt.xlabel('Time')
                    plt.ylabel('Weight')
                    plt.legend()
                    plt.grid(True)
                    
                    # Save line chart
                    line_weights_path = os.path.join(self.output_dir, "expert_weights_lines.png")
                    plt.savefig(line_weights_path)
                    plt.close()
                    
                    viz_paths.append(line_weights_path)
                    logger.info(f"Created expert weights line chart: {line_weights_path}")
            
            # 5. Weight concentration analysis
            if 'weight_concentration' in gating_data:
                concentration_data = gating_data['weight_concentration']
                
                if isinstance(concentration_data, dict) and 'gini_coefficients' in concentration_data:
                    gini_coefs = concentration_data['gini_coefficients']
                    timestamps = concentration_data.get('timestamps', list(range(len(gini_coefs))))
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(timestamps, gini_coefs, marker='o', linestyle='-', color='purple')
                    plt.title('Weight Concentration Over Time (Gini Coefficient)')
                    plt.xlabel('Time')
                    plt.ylabel('Gini Coefficient (higher = more concentrated)')
                    plt.grid(True)
                    
                    # Save concentration chart
                    concentration_path = os.path.join(self.output_dir, "weight_concentration.png")
                    plt.savefig(concentration_path)
                    plt.close()
                    
                    viz_paths.append(concentration_path)
                    logger.info(f"Created weight concentration chart: {concentration_path}")
            
            # 6. Decision boundary visualization if available
            if 'decision_boundaries' in gating_data:
                boundary_data = gating_data['decision_boundaries']
                
                if isinstance(boundary_data, dict) and 'features' in boundary_data and 'boundaries' in boundary_data:
                    features = boundary_data['features']
                    boundaries = boundary_data['boundaries']
                    
                    # If we have 2D decision boundaries
                    if len(features) == 2:
                        # Extract boundary points
                        x_points = [p[0] for p in boundaries]
                        y_points = [p[1] for p in boundaries]
                        labels = boundary_data.get('labels', ['Expert'] * len(boundaries))
                        
                        plt.figure(figsize=(12, 10))
                        scatter = plt.scatter(x_points, y_points, c=pd.factorize(labels)[0], 
                                              cmap='viridis', alpha=0.6, s=50)
                        
                        plt.title('Gating Network Decision Boundaries')
                        plt.xlabel(features[0])
                        plt.ylabel(features[1])
                        plt.colorbar(scatter, label='Expert')
                        plt.grid(True)
                        
                        # Add legend
                        unique_labels = list(set(labels))
                        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=plt.cm.viridis(pd.factorize(unique_labels)[0][i] / len(unique_labels)), 
                                            markersize=10) for i in range(len(unique_labels))]
                        plt.legend(handles, unique_labels, title='Expert', loc='upper right')
                        
                        # Save decision boundary chart
                        boundary_path = os.path.join(self.output_dir, "decision_boundaries.png")
                        plt.savefig(boundary_path)
                        plt.close()
                        
                        viz_paths.append(boundary_path)
                        logger.info(f"Created decision boundaries chart: {boundary_path}")
            
            return viz_paths
        
        except Exception as e:
            logger.error(f"Error creating gating visualizations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return viz_paths
    
    def create_overall_performance_visualizations(self):
        """
        Create visualizations for overall pipeline performance.
        
        Returns:
            List of paths to generated visualizations
        """
        if not self.checkpoint_data:
            logger.error("No checkpoint data loaded")
            return []
        
        viz_paths = []
        
        try:
            # Extract overall performance data
            performance_data = self.checkpoint_data.get('end_to_end_performance', {})
            
            if not performance_data:
                logger.warning("No end-to-end performance data found in checkpoint")
                return []
            
            # 1. Overall metrics visualization with color-coded significance
            metrics = performance_data.get('metrics', {})
            statistical_tests = performance_data.get('statistical_tests', {})
            
            if metrics:
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                
                # Check which metrics are statistically significant
                significant = [False] * len(metric_names)
                if statistical_tests and 'significance' in statistical_tests:
                    for i, metric in enumerate(metric_names):
                        if metric in statistical_tests['significance']:
                            significant[i] = statistical_tests['significance'][metric].get('result', False)
                
                # Create color palette based on significance
                colors = []
                for sig in significant:
                    colors.append('green' if sig else 'blue')
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(metric_names, metric_values, color=colors)
                
                plt.title('Overall Performance Metrics\n(Green = Statistically Significant)')
                plt.xlabel('Metric')
                plt.ylabel('Value')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.3f}', ha='center', va='bottom')
                
                # Save metrics chart
                metrics_path = os.path.join(self.output_dir, "overall_performance_metrics.png")
                plt.savefig(metrics_path)
                plt.close()
                
                viz_paths.append(metrics_path)
                logger.info(f"Created overall performance metrics chart: {metrics_path}")
            
            # 2. Temporal metrics visualization with confidence intervals
            temporal_metrics = performance_data.get('temporal_metrics', {})
            
            if temporal_metrics and 'timestamps' in temporal_metrics:
                # Extract data
                timestamps = temporal_metrics['timestamps']
                metrics_over_time = {
                    metric: temporal_metrics.get(metric, [0] * len(timestamps))
                    for metric in ['rmse', 'mae', 'r2'] if metric in temporal_metrics
                }
                
                # Get confidence intervals if available
                confidence_intervals = {}
                for metric in metrics_over_time:
                    ci_key = f"{metric}_ci"
                    if ci_key in temporal_metrics:
                        confidence_intervals[metric] = temporal_metrics[ci_key]
                
                if metrics_over_time:
                    # Create figure with multiple y-axes
                    fig, ax1 = plt.subplots(figsize=(14, 7))
                    
                    # Convert timestamps to datetime if they are strings
                    x = timestamps
                    if isinstance(timestamps[0], str):
                        try:
                            x = pd.to_datetime(timestamps)
                        except:
                            # If conversion fails, use numeric indices
                            x = range(len(timestamps))
                    
                    # Plot each metric
                    colors = sns.color_palette("Set1", len(metrics_over_time))
                    
                    for i, (metric, values) in enumerate(metrics_over_time.items()):
                        if i == 0:
                            ax = ax1
                            color = colors[i]
                        else:
                            # Create a twin axis for each additional metric
                            ax = ax1.twinx()
                            ax.spines['right'].set_position(('outward', 60 * (i-1)))
                            color = colors[i]
                        
                        # Plot the main line
                        line = ax.plot(x, values, marker='o', linestyle='-', color=color, label=metric.upper())
                        
                        # Add confidence intervals if available
                        if metric in confidence_intervals:
                            ci = confidence_intervals[metric]
                            if len(ci) == len(values) and isinstance(ci[0], (list, tuple)) and len(ci[0]) == 2:
                                lower = [c[0] for c in ci]
                                upper = [c[1] for c in ci]
                                ax.fill_between(x, lower, upper, color=color, alpha=0.2)
                        
                        ax.set_ylabel(metric.upper(), color=color)
                        ax.tick_params(axis='y', labelcolor=color)
                    
                    # Set title and labels
                    ax1.set_title('Performance Metrics Over Time')
                    ax1.set_xlabel('Time')
                    
                    # Create a combined legend
                    lines = []
                    labels = []
                    
                    for ax in fig.axes:
                        axline, axlabel = ax.get_legend_handles_labels()
                        lines.extend(axline)
                        labels.extend(axlabel)
                    
                    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.01),
                              fancybox=True, shadow=True, ncol=len(metrics_over_time))
                    
                    plt.tight_layout()
                    plt.subplots_adjust(bottom=0.15)
                    
                    # Save temporal metrics chart
                    temporal_path = os.path.join(self.output_dir, "performance_over_time.png")
                    plt.savefig(temporal_path)
                    plt.close(fig)
                    
                    viz_paths.append(temporal_path)
                    logger.info(f"Created performance over time chart: {temporal_path}")
            
            # 3. Baseline comparison visualization with statistical significance
            baseline_comparisons = performance_data.get('baseline_comparisons', {})
            
            if baseline_comparisons:
                # Extract RMSE values for each baseline and MoE
                baseline_names = list(baseline_comparisons.keys()) + ['MoE']
                rmse_values = [baseline.get('rmse', 0) for baseline in baseline_comparisons.values()]
                
                # Add MoE RMSE value
                moe_rmse = metrics.get('rmse', 0) if metrics else 0
                rmse_values.append(moe_rmse)
                
                # Get significance information if available
                is_significant = [False] * len(baseline_names)
                if 'significance_vs_baselines' in performance_data:
                    sig_data = performance_data['significance_vs_baselines']
                    for i, name in enumerate(baseline_names[:-1]):  # Skip MoE
                        if name in sig_data:
                            is_significant[i] = sig_data[name].get('is_significant', False)
                
                # Create color palette
                colors = sns.color_palette("muted", len(baseline_names))
                # Highlight MoE differently
                colors[-1] = 'darkred'  
                
                # Create bar chart
                plt.figure(figsize=(12, 6))
                bars = plt.bar(baseline_names, rmse_values, color=colors)
                
                # Add significance markers
                for i, (bar, sig) in enumerate(zip(bars[:-1], is_significant[:-1])):
                    if sig:
                        plt.text(bar.get_x() + bar.get_width()/2., 0.05,
                                '*', ha='center', va='bottom', 
                                fontsize=16, color='red')
                
                plt.title('RMSE Comparison: MoE vs Baselines\n* = Statistically Significant Difference')
                plt.xlabel('Model')
                plt.ylabel('RMSE (lower is better)')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.3f}', ha='center', va='bottom')
                
                # Save baseline comparison chart
                baseline_path = os.path.join(self.output_dir, "baseline_comparison.png")
                plt.savefig(baseline_path)
                plt.close()
                
                viz_paths.append(baseline_path)
                logger.info(f"Created baseline comparison chart: {baseline_path}")
            
            # 4. Statistical tests visualization
            if statistical_tests:
                # Create a summary table of statistical tests
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.axis('tight')
                ax.axis('off')
                
                # Prepare table data
                table_data = [['Test Name', 'Statistic', 'p-value', 'Result', 'Interpretation']]
                
                for test_name, test_data in statistical_tests.items():
                    if isinstance(test_data, dict):
                        statistic = test_data.get('statistic', 'N/A')
                        p_value = test_data.get('p_value', 'N/A')
                        result = test_data.get('result', False)
                        
                        # Generate interpretation
                        if test_name == 'normality':
                            interpretation = 'Data is normally distributed' if result else 'Data is not normally distributed'
                        elif test_name == 'significance':
                            interpretation = 'Results are statistically significant' if result else 'Results are not statistically significant'
                        else:
                            interpretation = 'Test passed' if result else 'Test failed'
                        
                        # Format p-value
                        if isinstance(p_value, (int, float)):
                            p_value_str = f'{p_value:.4f}'
                        else:
                            p_value_str = str(p_value)
                        
                        # Format statistic
                        if isinstance(statistic, (int, float)):
                            statistic_str = f'{statistic:.4f}'
                        else:
                            statistic_str = str(statistic)
                        
                        table_data.append([test_name.title(), statistic_str, p_value_str, 
                                        'Passed' if result else 'Failed', interpretation])
                
                # Create table
                table = ax.table(cellText=table_data, colWidths=[0.15, 0.15, 0.15, 0.15, 0.4], 
                             loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                # Style the header row - using direct indices to avoid slice indexing issues
                for j in range(len(table_data[0])):
                    cell = table[(0, j)]
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(color='white', fontweight='bold')
                
                # Style result cells
                for i in range(1, len(table_data)):
                    # Color pass/fail cells
                    try:
                        result_cell = table[(i, 3)]
                        if table_data[i][3] == 'Passed':
                            result_cell.set_facecolor('lightgreen')
                        else:
                            result_cell.set_facecolor('lightpink')
                    except KeyError:
                        # Skip if the cell doesn't exist
                        pass
                        
                plt.title('Statistical Test Results', fontsize=14, pad=20)
                plt.tight_layout()
                
                # Save statistical tests table
                stats_path = os.path.join(self.output_dir, "statistical_tests.png")
                plt.savefig(stats_path, bbox_inches='tight')
                plt.close()
                
                viz_paths.append(stats_path)
                logger.info(f"Created statistical tests visualization: {stats_path}")
            
            # 5. Confidence calibration plot
            confidence_metrics = performance_data.get('confidence_metrics', {})
            
            if confidence_metrics and 'bins' in confidence_metrics and 'calibration' in confidence_metrics:
                bins = confidence_metrics['bins']
                calibration = confidence_metrics['calibration']
                
                plt.figure(figsize=(10, 10))
                
                # Plot the calibration curve
                plt.plot(bins, calibration, 'bo-', label='Model calibration')
                
                # Plot the ideal calibration (diagonal line)
                plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                
                # Add the calibration error if available
                if 'calibration_error' in confidence_metrics:
                    calib_error = confidence_metrics['calibration_error']
                    if isinstance(calib_error, (int, float)):
                        plt.text(0.05, 0.95, f"Calibration Error: {calib_error:.3f}", 
                                ha='left', va='top', transform=plt.gca().transAxes,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    else:
                        plt.text(0.05, 0.95, f"Calibration Error: {calib_error}", 
                                ha='left', va='top', transform=plt.gca().transAxes,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.xlabel('Predicted Probability')
                plt.ylabel('True Probability')
                plt.title('Calibration Curve')
                plt.legend(loc='lower right')
                plt.grid(True)
                
                # Save calibration plot
                calibration_path = os.path.join(self.output_dir, "confidence_calibration.png")
                plt.savefig(calibration_path)
                plt.close()
                
                viz_paths.append(calibration_path)
                logger.info(f"Created confidence calibration plot: {calibration_path}")
            
            return viz_paths
        
        except Exception as e:
            logger.error(f"Error creating overall performance visualizations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return viz_paths
    
    def create_all_visualizations(self):
        """
        Create all available visualizations from checkpoint data.
        
        Returns:
            Dictionary of visualization paths categorized by type
        """
        if not self.checkpoint_data:
            logger.error("No checkpoint data loaded")
            return {}
        
        visualizations = {
            'expert_benchmarks': self.create_expert_benchmarks(),
            'gating_network': self.create_gating_visualizations(),
            'overall_performance': self.create_overall_performance_visualizations()
        }
        
        # Create a summary HTML page
        self.create_summary_html(visualizations)
        
        return visualizations
    
    def create_summary_html(self, visualizations):
        """
        Create a summary HTML page with all visualizations.
        
        Args:
            visualizations: Dictionary of visualization paths by category
        
        Returns:
            Path to the generated HTML file
        """
        html_path = os.path.join(self.output_dir, "visualization_summary.html")
        
        # Extract checkpoint name or use a default
        checkpoint_name = self.checkpoint_data.get('name', 'MoE Checkpoint') if self.checkpoint_data else 'MoE Checkpoint'
        timestamp = self.checkpoint_data.get('timestamp', datetime.now().isoformat()) if self.checkpoint_data else datetime.now().isoformat()
        
        try:
            with open(html_path, 'w') as f:
                # Write HTML header
                f.write(f'''<!DOCTYPE html>
<html>
<head>
    <title>MoE Visualization Summary - {checkpoint_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .visualization {{ margin: 20px 0; }}
        .visualization img {{ max-width: 100%; border: 1px solid #eee; }}
        .footer {{ margin-top: 30px; text-align: center; font-size: 0.8em; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MoE Visualization Summary</h1>
        <p>Checkpoint: <strong>{checkpoint_name}</strong></p>
        <p>Generated: <strong>{timestamp}</strong></p>
''')
                
                # Expert Benchmarks Section
                f.write('''
        <div class="section">
            <h2>Expert Model Benchmarks</h2>
''')
                for viz_path in visualizations.get('expert_benchmarks', []):
                    viz_filename = os.path.basename(viz_path)
                    f.write(f'''
            <div class="visualization">
                <h3>{viz_filename.replace('.png', '').replace('_', ' ').title()}</h3>
                <img src="{viz_filename}" alt="{viz_filename}" />
            </div>
''')
                f.write('        </div>\n')
                
                # Gating Network Section
                f.write('''
        <div class="section">
            <h2>Gating Network Analysis</h2>
''')
                for viz_path in visualizations.get('gating_network', []):
                    viz_filename = os.path.basename(viz_path)
                    f.write(f'''
            <div class="visualization">
                <h3>{viz_filename.replace('.png', '').replace('_', ' ').title()}</h3>
                <img src="{viz_filename}" alt="{viz_filename}" />
            </div>
''')
                f.write('        </div>\n')
                
                # Architecture Flow Section
                f.write('''
        <div class="section">
            <h2>Architecture Flow</h2>
''')
                for viz_path in visualizations.get('architecture_flow', []):
                    viz_filename = os.path.basename(viz_path)
                    f.write(f'''
            <div class="visualization">
                <h3>{viz_filename.replace('.png', '').replace('_', ' ').title()}</h3>
                <img src="{viz_filename}" alt="{viz_filename}" />
            </div>
''')
                f.write('        </div>\n')

                # Overall Performance Section
                f.write('''
        <div class="section">
            <h2>Overall Performance Metrics</h2>
''')
                for viz_path in visualizations.get('overall_performance', []):
                    viz_filename = os.path.basename(viz_path)
                    f.write(f'''
            <div class="visualization">
                <h3>{viz_filename.replace('.png', '').replace('_', ' ').title()}</h3>
                <img src="{viz_filename}" alt="{viz_filename}" />
            </div>
''')
                f.write('        </div>\n')
                
                # Footer
                f.write('''
        <div class="footer">
            <p>Generated by MoE Visualizer - Temporary directory: {}</p>
        </div>
    </div>
</body>
</html>
'''.format(self.output_dir))
            
            logger.info(f"Created visualization summary HTML: {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Error creating summary HTML: {str(e)}")
            return None
    
    def open_visualization_in_browser(self):
        """Open the visualization summary in the default web browser."""
        html_path = os.path.join(self.output_dir, "visualization_summary.html")
        if os.path.exists(html_path):
            import webbrowser
            webbrowser.open(f"file://{html_path}")
            return True
        else:
            logger.error(f"Visualization summary HTML not found: {html_path}")
            return False

    def _generate_architecture_flow(self):
        """Generate architecture flow visualization showing component connections."""
        G = nx.DiGraph()
        
        # Add nodes for each component
        components = {
            "data_connector": "Data Connector",
            "data_quality": "Data Quality",
            "expert_1": "Expert 1",
            "expert_2": "Expert 2",
            "expert_3": "Expert 3",
            "gating": "Gating Network",
            "integration": "Integration Layer",
            "output": "Output"
        }
        
        # Add nodes with positions
        pos = {
            "data_connector": (0, 0.5),
            "data_quality": (1, 0.5),
            "expert_1": (2, 0.8),
            "expert_2": (2, 0.5),
            "expert_3": (2, 0.2),
            "gating": (2, -0.2),
            "integration": (3, 0.5),
            "output": (4, 0.5)
        }
        
        for node_id, label in components.items():
            color = self.component_colors[node_id.split('_')[0]]
            G.add_node(node_id, label=label, color=color)
        
        # Add edges
        edges = [
            ("data_connector", "data_quality"),
            ("data_quality", "expert_1"),
            ("data_quality", "expert_2"),
            ("data_quality", "expert_3"),
            ("data_quality", "gating"),
            ("expert_1", "integration"),
            ("expert_2", "integration"),
            ("expert_3", "integration"),
            ("gating", "integration"),
            ("integration", "output")
        ]
        G.add_edges_from(edges)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Draw nodes
        for node in G.nodes():
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[node],
                node_color=[G.nodes[node]["color"]],
                node_size=2000,
                alpha=0.7
            )
        
        # Draw edges with arrows
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20
        )
        
        # Add labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title("MoE Architecture Flow", pad=20, size=14)
        plt.axis('off')
        
        # Save visualization
        viz_path = os.path.join(self.output_dir, "architecture_flow.png")
        plt.savefig(viz_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.visualizations["architecture_flow"].append(viz_path)

    def generate_visualizations(self):
        """Generate all visualizations for the MoE framework."""
        self.create_expert_benchmarks()
        self.create_gating_visualizations()
        self.create_overall_performance_visualizations()
        self._generate_architecture_flow()
        self.create_summary_html(self.visualizations)
