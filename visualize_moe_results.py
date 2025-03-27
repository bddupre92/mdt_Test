#!/usr/bin/env python
"""
MoE Visualization Script

This script generates visualizations for MoE pipeline results using the most recent checkpoint.
All visualizations are saved to a temporary directory which can be easily cleaned up.
"""

import os
import sys
import glob
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Import the MoE visualizer
from visualization.moe_visualizer import MoEVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_latest_checkpoint(checkpoint_dir=None):
    """
    Find the most recent checkpoint file.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints, uses default paths if None
        
    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    if checkpoint_dir is None:
        # Default checkpoint directories to check
        checkpoint_dirs = [
            'results/moe_run/dev/checkpoints',
            'results/moe_run/checkpoints',
            'results/checkpoints'
        ]
    else:
        checkpoint_dirs = [checkpoint_dir]
    
    # Find all checkpoint files across specified directories
    all_checkpoints = []
    for directory in checkpoint_dirs:
        if os.path.exists(directory):
            json_files = glob.glob(os.path.join(directory, '*.json'))
            all_checkpoints.extend(json_files)
    
    if not all_checkpoints:
        logger.error("No checkpoint files found in specified directories")
        return None
    
    # Find the most recent file by modification time
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    
    return latest_checkpoint

def main():
    """Generate visualizations for MoE results using the latest checkpoint."""
    parser = argparse.ArgumentParser(description='Generate visualizations for MoE pipeline results')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file (uses latest if not specified)')
    parser.add_argument('--output-dir', type=str, help='Custom output directory (uses temp dir if not specified)')
    parser.add_argument('--no-temp', action='store_true', help='Do not use temporary directory')
    parser.add_argument('--browser', action='store_true', help='Open visualization summary in browser')
    
    args = parser.parse_args()
    
    # Find the latest checkpoint if not specified
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            logger.error("No checkpoint file found. Please specify a checkpoint with --checkpoint")
            return 1
    
    # Create visualizer
    visualizer = MoEVisualizer(
        checkpoint_path=checkpoint_path,
        use_temp_dir=not args.no_temp,
        output_dir=args.output_dir
    )
    
    # Generate all visualizations
    logger.info("Generating visualizations...")
    visualizations = visualizer.create_all_visualizations()
    
    # Count total visualizations created
    total_visualizations = sum(len(viz_list) for viz_list in visualizations.values())
    logger.info(f"Created {total_visualizations} visualizations in: {visualizer.output_dir}")
    
    # Print summary of created visualizations
    for category, viz_list in visualizations.items():
        if viz_list:
            logger.info(f"{category.replace('_', ' ').title()} visualizations: {len(viz_list)}")
    
    # Open in browser if requested
    if args.browser:
        logger.info("Opening visualization summary in browser...")
        visualizer.open_visualization_in_browser()
    
    logger.info(f"Visualization complete. All files saved to: {visualizer.output_dir}")
    logger.info("Note: If using a temporary directory, files will be deleted when the system is restarted.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
