"""
MoE Architecture Visualization Module - Part 2 (Gating Network & Integration Flow)
"""

def visualize_domain_specific_metrics(self):
    """Visualize domain-specific metrics for each expert type."""
    if not self.checkpoint_data:
        logger.error("No checkpoint data loaded")
        return None
    
    try:
        experts_data = self.checkpoint_data.get("experts", {})
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
                self._visualize_physiological_metrics(ax, experts, experts_data)
            elif expert_type == "environmental":
                self._visualize_environmental_metrics(ax, experts, experts_data)
            elif expert_type == "behavioral":
                self._visualize_behavioral_metrics(ax, experts, experts_data)
            elif expert_type == "medication_history":
                self._visualize_medication_metrics(ax, experts, experts_data)
            else:
                ax.text(0.5, 0.5, f"No specific metrics for {expert_type} type",
                       ha='center', va='center', fontsize=12)
                ax.set_title(f"{expert_type.capitalize()} Expert Metrics", fontsize=14)
                ax.axis('off')
        
        plt.suptitle("Domain-Specific Expert Metrics", fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, "domain_specific_metrics.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created domain-specific metrics visualization: {output_path}")
        self.visualizations["expert_models"].append({
            "title": "Domain-Specific Expert Metrics",
            "path": output_path,
            "description": "Specialized metrics for each expert domain"
        })
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating domain-specific metrics visualization: {str(e)}")
        return None

def _visualize_physiological_metrics(self, ax, experts, experts_data):
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
              color=self.expert_colors['physiological'],
              alpha=0.5 + 0.5 * (i / len(experts)))
    
    # Add labels and legend
    ax.set_xticks(positions + bar_width * (len(experts) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title('Physiological Expert Metrics', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def _visualize_environmental_metrics(self, ax, experts, experts_data):
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
               color=self.expert_colors['environmental'],
               alpha=0.7)
        ax.fill(angles, values_list, alpha=0.1, 
                color=self.expert_colors['environmental'])
    
    ax.set_ylim(0, 1)
    ax.set_title('Environmental Expert Metrics', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=8)

def _visualize_behavioral_metrics(self, ax, experts, experts_data):
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
               color=self.expert_colors['behavioral'],
               alpha=0.5 + 0.5 * (i / len(experts)))
    
    # Add labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Score')
    ax.set_title('Behavioral Expert Metrics', fontsize=14)
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

def _visualize_medication_metrics(self, ax, experts, experts_data):
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
              color=self.expert_colors['medication_history'],
              alpha=0.5 + 0.5 * (i / len(experts)))
    
    # Add labels
    ax.set_xticks(positions + bar_width * (len(experts) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title('Medication Expert Metrics', fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def visualize_gating_mechanism(self):
    """Visualize the gating mechanism of the MoE framework."""
    if not self.checkpoint_data:
        logger.error("No checkpoint data loaded")
        return None
    
    try:
        # Extract gating network information
        gating_info = self.checkpoint_data.get("gating_network", {})
        gating_type = gating_info.get("type", "unknown")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create gating network diagram based on type
        if gating_type == "quality_aware":
            self._visualize_quality_aware_gating(ax, gating_info)
        elif gating_type == "meta_learner":
            self._visualize_meta_learner_gating(ax, gating_info)
        else:
            self._visualize_standard_gating(ax, gating_info)
        
        # Add title
        plt.suptitle(f"MoE Gating Network: {gating_type.replace('_', ' ').title()}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save visualization
        output_path = os.path.join(self.output_dir, "gating_mechanism.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created gating mechanism visualization: {output_path}")
        self.visualizations["gating_network"].append({
            "title": "Gating Network Mechanism",
            "path": output_path,
            "description": f"Visualization of the {gating_type.replace('_', ' ').title()} gating mechanism"
        })
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating gating mechanism visualization: {str(e)}")
        return None

def _visualize_standard_gating(self, ax, gating_info):
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
    expert_colors = list(self.expert_colors.values())
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

def _visualize_quality_aware_gating(self, ax, gating_info):
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
    expert_colors = list(self.expert_colors.values())
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

def _visualize_meta_learner_gating(self, ax, gating_info):
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
    expert_colors = list(self.expert_colors.values())
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
