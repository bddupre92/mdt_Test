"""
Test script for MoE Visualizer
"""

import os
from visualization.moe_visualizer import MoEVisualizer

def main():
    # Get the checkpoint path
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'test_checkpoint.json')
    
    # Initialize visualizer with checkpoint and temporary directory
    visualizer = MoEVisualizer(checkpoint_path=checkpoint_path, use_temp_dir=True)
    
    # Generate visualizations
    visualizer.generate_visualizations()
    
    # Open in browser
    visualizer.open_visualization_in_browser()

if __name__ == "__main__":
    main()
