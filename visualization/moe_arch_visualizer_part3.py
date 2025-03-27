"""
MoE Architecture Visualization Module - Part 3 (Architecture & Integration Flow)
"""

def visualize_architecture_diagram(self):
    """Create a comprehensive architecture diagram of the MoE framework."""
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Define component coordinates
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
        self._draw_component(ax, components['data_input'], 0.1, 0.05, 'Data Input', 'lightblue')
        self._draw_component(ax, components['preprocessing'], 0.15, 0.05, 'Preprocessing\nManager', 'lightblue')
        
        # Expert layer components
        self._draw_component(ax, components['physiological_expert'], 0.1, 0.08, 'Physiological\nExpert', self.expert_colors['physiological'])
        self._draw_component(ax, components['environmental_expert'], 0.1, 0.08, 'Environmental\nExpert', self.expert_colors['environmental'])
        self._draw_component(ax, components['behavioral_expert'], 0.1, 0.08, 'Behavioral\nExpert', self.expert_colors['behavioral'])
        self._draw_component(ax, components['medication_expert'], 0.1, 0.08, 'Medication\nExpert', self.expert_colors['medication_history'])
        
        # Gating network
        self._draw_component(ax, components['gating_network'], 0.2, 0.08, 'Gating Network', 'lightgreen')
        
        # Integration layer
        self._draw_component(ax, components['integration_connector'], 0.2, 0.05, 'Integration Connector', 'orange')
        
        # Output layer
        self._draw_component(ax, components['prediction_output'], 0.1, 0.05, 'Prediction Output', 'lightgray')
        
        # Management components
        self._draw_component(ax, components['state_manager'], 0.1, 0.05, 'State Manager', 'lightgray')
        self._draw_component(ax, components['event_system'], 0.1, 0.05, 'Event System', 'lightgray')
        self._draw_component(ax, components['meta_learner'], 0.1, 0.05, 'Meta Learner', 'purple')
        
        # Draw connections
        # Data flow connections
        self._draw_arrow(ax, components['data_input'], components['preprocessing'])
        self._draw_arrow(ax, components['preprocessing'], components['physiological_expert'])
        self._draw_arrow(ax, components['preprocessing'], components['environmental_expert'])
        self._draw_arrow(ax, components['preprocessing'], components['behavioral_expert'])
        self._draw_arrow(ax, components['preprocessing'], components['medication_expert'])
        
        # Expert to gating connections
        self._draw_arrow(ax, components['physiological_expert'], components['gating_network'])
        self._draw_arrow(ax, components['environmental_expert'], components['gating_network'])
        self._draw_arrow(ax, components['behavioral_expert'], components['gating_network'])
        self._draw_arrow(ax, components['medication_expert'], components['gating_network'])
        
        # Meta-learner connection
        self._draw_arrow(ax, components['meta_learner'], components['gating_network'])
        
        # Gating to integration
        self._draw_arrow(ax, components['gating_network'], components['integration_connector'])
        
        # Integration to output
        self._draw_arrow(ax, components['integration_connector'], components['prediction_output'])
        
        # State manager connections
        self._draw_dashed_arrow(ax, components['state_manager'], components['gating_network'])
        self._draw_dashed_arrow(ax, components['state_manager'], components['integration_connector'])
        
        # Event system connections
        self._draw_dashed_arrow(ax, components['event_system'], components['physiological_expert'])
        self._draw_dashed_arrow(ax, components['event_system'], components['environmental_expert'])
        self._draw_dashed_arrow(ax, components['event_system'], components['behavioral_expert'])
        self._draw_dashed_arrow(ax, components['event_system'], components['medication_expert'])
        self._draw_dashed_arrow(ax, components['event_system'], components['gating_network'])
        
        # Add title and legend
        plt.suptitle("MoE Framework Architecture", fontsize=18, y=0.98)
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Data Layer'),
            mpatches.Patch(color=self.expert_colors['physiological'], label='Physiological Expert'),
            mpatches.Patch(color=self.expert_colors['environmental'], label='Environmental Expert'),
            mpatches.Patch(color=self.expert_colors['behavioral'], label='Behavioral Expert'),
            mpatches.Patch(color=self.expert_colors['medication_history'], label='Medication Expert'),
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
        output_path = os.path.join(self.output_dir, "architecture_diagram.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created architecture diagram: {output_path}")
        self.visualizations["integration_flow"].append({
            "title": "MoE Architecture Diagram",
            "path": output_path,
            "description": "Comprehensive diagram of the MoE framework architecture"
        })
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating architecture diagram: {str(e)}")
        return None

def _draw_component(self, ax, position, width, height, label, color):
    """Helper to draw a component box with label."""
    rect = plt.Rectangle((position[0] - width/2, position[1] - height/2), 
                          width, height, facecolor=color, alpha=0.8, 
                          edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(position[0], position[1], label, ha='center', va='center', 
           fontsize=10, fontweight='bold')

def _draw_arrow(self, ax, start_pos, end_pos):
    """Helper to draw an arrow between components."""
    ax.arrow(start_pos[0], start_pos[1], 
            end_pos[0] - start_pos[0], end_pos[1] - start_pos[1],
            head_width=0.01, head_length=0.01, fc='black', ec='black',
            length_includes_head=True)

def _draw_dashed_arrow(self, ax, start_pos, end_pos):
    """Helper to draw a dashed arrow between components."""
    ax.annotate("", xy=end_pos, xytext=start_pos,
               arrowprops=dict(arrowstyle="->", linestyle='dashed'))

def visualize_data_flow(self):
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
                               facecolor=self.expert_colors.get(expert_type.lower(), '#CCCCCC'),
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
        output_path = os.path.join(self.output_dir, "data_flow.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created data flow visualization: {output_path}")
        self.visualizations["integration_flow"].append({
            "title": "MoE Pipeline Data Flow",
            "path": output_path,
            "description": "Visualization of data flow through the MoE pipeline"
        })
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating data flow visualization: {str(e)}")
        return None
