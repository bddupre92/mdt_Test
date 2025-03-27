"""
MoE Architecture Visualization Module - Part 4 (Performance & HTML Generator)
"""

def visualize_overall_performance(self):
    """Visualize the overall performance metrics of the MoE system."""
    if not self.checkpoint_data:
        logger.error("No checkpoint data loaded")
        return None
    
    try:
        # Extract performance metrics
        performance = self.checkpoint_data.get("performance", {})
        overall_metrics = performance.get("overall_metrics", {})
        
        if not overall_metrics:
            logger.warning("No overall performance metrics found in checkpoint")
            return None
        
        # Extract metrics
        metrics = {
            "RMSE": overall_metrics.get("rmse", 0),
            "MAE": overall_metrics.get("mae", 0),
            "R²": overall_metrics.get("r2", 0)
        }
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart of main metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # For better visualization, scale R² differently (it's from 0-1 while errors can be larger)
        scaled_values = [
            v if k != "R²" else v * max(metrics["RMSE"], metrics["MAE"]) * 1.2
            for k, v in zip(metric_names, metric_values)
        ]
        
        colors = ['#FF9999', '#66B2FF', '#99CC99']
        bars = ax1.bar(metric_names, scaled_values, color=colors, alpha=0.8)
        
        # Add actual values as text on bars
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f"{value:.4f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add labels
        ax1.set_title('Overall Performance Metrics', fontsize=14)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add a note about R² scaling
        ax1.text(0.95, 0.05, "Note: R² is scaled for visualization",
                ha='right', va='bottom', transform=ax1.transAxes,
                fontsize=8, fontStyle='italic')
        
        # Gauge chart for overall score
        # Calculate an overall score (weighted average of normalized metrics)
        rmse_norm = 1 - min(1, metrics["RMSE"] / 10)  # Normalize RMSE (lower is better)
        mae_norm = 1 - min(1, metrics["MAE"] / 5)     # Normalize MAE (lower is better)
        r2_norm = max(0, metrics["R²"])               # R² is already 0-1 (higher is better)
        
        overall_score = (rmse_norm * 0.3) + (mae_norm * 0.3) + (r2_norm * 0.4)
        overall_score = min(1, max(0, overall_score))  # Ensure between 0-1
        
        # Create gauge chart
        gauge_theta = np.linspace(0, 180, 100)
        gauge_r = np.ones_like(gauge_theta)
        
        # Color map for gauge
        cmap = LinearSegmentedColormap.from_list('gauge_cmap', ['#FF0000', '#FFFF00', '#00FF00'])
        
        # Plot the gauge background
        ax2.plot(gauge_theta, gauge_r, color='lightgray', linewidth=15, solid_capstyle='round')
        
        # Calculate the position for the score
        score_theta = overall_score * 180
        
        # Plot the filled gauge up to the score
        mask = gauge_theta <= score_theta
        gauge_colors = cmap(gauge_theta / 180)
        
        for i in range(len(gauge_theta) - 1):
            if mask[i]:
                ax2.plot(gauge_theta[i:i+2], gauge_r[i:i+2], color=gauge_colors[i], 
                        linewidth=15, solid_capstyle='round')
        
        # Set the gauge appearance
        ax2.set_theta_zero_location("S")
        ax2.set_theta_direction(-1)
        ax2.set_thetamin(0)
        ax2.set_thetamax(180)
        
        # Remove the radial ticks and labels
        ax2.set_rticks([])
        ax2.set_xticks([0, 45, 90, 135, 180])
        ax2.set_xticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
        
        # Add titles and score
        ax2.set_title('Overall System Score', fontsize=14)
        ax2.text(0, -0.15, f"{overall_score:.2f}", ha='center', va='center', 
                fontsize=20, fontweight='bold', transform=ax2.transAxes)
        
        # Add score labels
        ax2.text(-0.2, -0.3, "Poor", ha='left', fontsize=10, transform=ax2.transAxes)
        ax2.text(0, -0.3, "Average", ha='center', fontsize=10, transform=ax2.transAxes)
        ax2.text(0.2, -0.3, "Excellent", ha='right', fontsize=10, transform=ax2.transAxes)
        
        # Add formula explanation
        formula_text = (f"Overall Score = (RMSE_norm × 0.3) + (MAE_norm × 0.3) + (R² × 0.4)\n"
                       f"= ({rmse_norm:.2f} × 0.3) + ({mae_norm:.2f} × 0.3) + ({r2_norm:.2f} × 0.4)")
        
        ax2.text(0, -0.4, formula_text, ha='center', va='center', fontsize=9,
                transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.suptitle('MoE Framework Performance Summary', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save visualization
        output_path = os.path.join(self.output_dir, "overall_performance.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created overall performance visualization: {output_path}")
        self.visualizations["performance"].append({
            "title": "Overall Performance Metrics",
            "path": output_path,
            "description": "Summary of key performance metrics for the MoE framework"
        })
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating overall performance visualization: {str(e)}")
        return None

def visualize_component_contribution(self):
    """Visualize the contribution of each component to the final prediction."""
    if not self.checkpoint_data:
        logger.error("No checkpoint data loaded")
        return None
    
    try:
        # Extract expert weights and performance info
        experts_data = self.checkpoint_data.get("experts", {})
        gating_info = self.checkpoint_data.get("gating_network", {})
        performance = self.checkpoint_data.get("performance", {})
        
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
        expert_types = {expert_id: experts_data[expert_id].get("type", "unknown") for expert_id in expert_ids}
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Create pie chart of expert weights
        labels = [f"Expert {expert_id}\n({expert_types[expert_id]})" for expert_id in expert_ids]
        sizes = [weights[expert_id] for expert_id in expert_ids]
        colors = [self.expert_colors.get(expert_types[expert_id], "#CCCCCC") for expert_id in expert_ids]
        
        # Create pie chart with expert contribution
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=None, autopct='%1.1f%%', startangle=90,
            colors=colors, wedgeprops=dict(width=0.5, edgecolor='w')
        )
        
        # Add a legend
        ax1.legend(wedges, labels, title="Expert Contribution", 
                 loc="center left", bbox_to_anchor=(0.9, 0, 0.5, 1))
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.set_aspect('equal')
        ax1.set_title('Expert Contribution to MoE Prediction', fontsize=14)
        
        # Performance improvement visualization
        # Create horizontal bar chart showing the relative performance of MoE vs best expert
        
        # Get MoE and individual expert performances
        moe_rmse = performance.get("overall_metrics", {}).get("rmse", 0)
        expert_rmse = {expert_id: experts_data[expert_id].get("rmse", 0) for expert_id in expert_ids}
        
        # Find the best expert (lowest RMSE)
        best_expert_id = min(expert_rmse, key=expert_rmse.get)
        best_expert_rmse = expert_rmse[best_expert_id]
        
        # Calculate improvement percentage
        if best_expert_rmse > 0:
            improvement = (best_expert_rmse - moe_rmse) / best_expert_rmse * 100
        else:
            improvement = 0
            
        # Prepare data for horizontal bar chart
        system_names = ['MoE Framework', f'Best Expert\n({best_expert_id})']
        rmse_values = [moe_rmse, best_expert_rmse]
        
        # Create horizontal bar chart
        bars = ax2.barh(system_names, rmse_values, color=['#6699CC', '#CC6666'])
        
        # Add RMSE values on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                    f"RMSE: {width:.4f}", ha='left', va='center', fontsize=10)
        
        # Add improvement callout
        if improvement > 0:
            ax2.annotate(
                f"{improvement:.1f}% improvement",
                xy=(moe_rmse, 0),
                xytext=(moe_rmse + (best_expert_rmse - moe_rmse)/2, 0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='green'),
                color='green', fontweight='bold', fontsize=12, ha='center'
            )
        
        # Add labels and title
        ax2.set_title('Performance Comparison', fontsize=14)
        ax2.set_xlabel('RMSE (lower is better)', fontsize=12)
        ax2.invert_yaxis()  # Invert to match pie chart order
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.suptitle('MoE Component Contribution Analysis', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save visualization
        output_path = os.path.join(self.output_dir, "component_contribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created component contribution visualization: {output_path}")
        self.visualizations["performance"].append({
            "title": "Component Contribution Analysis",
            "path": output_path,
            "description": "Analysis of each expert's contribution to the final prediction"
        })
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating component contribution visualization: {str(e)}")
        return None

def create_visualization_summary(self):
    """Create an HTML summary of all generated visualizations."""
    try:
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MoE Architecture Visualization</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9f9f9;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #2c3e50;
                    color: white;
                    border-radius: 5px;
                }}
                .section {{
                    margin-bottom: 40px;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .visualization {{
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid #eee;
                }}
                .visualization:last-child {{
                    border-bottom: none;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                }}
                .description {{
                    margin-top: 15px;
                    font-style: italic;
                    color: #666;
                }}
                .navigation {{
                    position: sticky;
                    top: 20px;
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .navigation ul {{
                    list-style-type: none;
                    padding: 0;
                }}
                .navigation li {{
                    margin-bottom: 10px;
                }}
                .navigation a {{
                    text-decoration: none;
                    color: #3498db;
                }}
                .navigation a:hover {{
                    text-decoration: underline;
                }}
                @media (max-width: 768px) {{
                    .navigation {{
                        position: static;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MoE Architecture Visualization</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Checkpoint: {os.path.basename(self.checkpoint_path) if self.checkpoint_path else 'None'}</p>
            </div>

            <div class="navigation">
                <h3>Navigation</h3>
                <ul>
                    <li><a href="#expert-models">Expert Models</a></li>
                    <li><a href="#gating-network">Gating Network</a></li>
                    <li><a href="#integration-flow">Integration Flow</a></li>
                    <li><a href="#performance">Performance Analysis</a></li>
                </ul>
            </div>
        """
        
        # Add sections for each visualization type
        sections = [
            ("expert-models", "Expert Models", self.visualizations["expert_models"]),
            ("gating-network", "Gating Network", self.visualizations["gating_network"]),
            ("integration-flow", "Integration Flow", self.visualizations["integration_flow"]),
            ("performance", "Performance Analysis", self.visualizations["performance"])
        ]
        
        for section_id, section_name, visualizations in sections:
            if visualizations:
                html_content += f"""
                <div class="section" id="{section_id}">
                    <h2>{section_name}</h2>
                """
                
                for viz in visualizations:
                    # Get relative path for HTML
                    rel_path = os.path.basename(viz["path"])
                    
                    html_content += f"""
                    <div class="visualization">
                        <h3>{viz["title"]}</h3>
                        <img src="{rel_path}" alt="{viz["title"]}">
                        <p class="description">{viz["description"]}</p>
                    </div>
                    """
                
                html_content += """
                </div>
                """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML file
        output_path = os.path.join(self.output_dir, "visualization_summary.html")
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created visualization summary HTML: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating visualization summary: {str(e)}")
        return None
