#!/usr/bin/env python3
"""
Run MoE Validation with Enhanced Synthetic Data

This script prepares enhanced synthetic data and runs the MoE validation framework with it.
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run MoE validation with enhanced synthetic data")
    parser.add_argument("--enhanced-data-dir", default="./test_data/enhanced_output",
                        help="Directory containing enhanced synthetic data")
    parser.add_argument("--results-dir", default="./results/moe_validation",
                        help="Directory to store MoE validation results")
    parser.add_argument("--drift-type", default="all",
                        choices=["none", "sudden", "gradual", "recurring", "all"],
                        help="Type of drift to include in the validation")
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive visualization")
    parser.add_argument("--explainers", default="shap,feature_importance",
                        help="Comma-separated list of explainers to use")
    parser.add_argument("--notify", action="store_true",
                        help="Enable drift notifications")
    parser.add_argument("--notify-threshold", type=float, default=0.5,
                        help="Threshold for drift notifications")
    
    args = parser.parse_args()
    
    # Prepare enhanced data for MoE validation
    logger.info("Step 1: Preparing enhanced synthetic data for MoE validation")
    
    # Import the prepare_enhanced_data function
    try:
        from prepare_enhanced_validation import prepare_enhanced_data
        
        # Prepare enhanced data
        prepare_args = {
            'enhanced_data_dir': args.enhanced_data_dir,
            'results_dir': args.results_dir,
            'drift_type': args.drift_type
        }
        
        result = prepare_enhanced_data(prepare_args)
        
        if not result.get('success', False):
            logger.error(f"Failed to prepare enhanced data: {result.get('message', 'Unknown error')}")
            return 1
        
        logger.info(f"Successfully prepared enhanced data: {result.get('message', '')}")
        config_path = result.get('config_path')
        
    except ImportError:
        logger.error("Failed to import prepare_enhanced_validation module")
        return 1
    
    # Run MoE validation with enhanced data
    logger.info("Step 2: Running MoE validation with enhanced data")
    
    # Set environment variables to pass configuration
    os.environ['ENHANCED_DATA_CONFIG'] = config_path
    logger.info(f"Setting ENHANCED_DATA_CONFIG environment variable to {config_path}")
    
    # Build command without the enhanced-data-config parameter
    cmd = [
        "python", "main_v2.py", "moe_validation"
    ]
    
    # Add interactive flag if requested
    if args.interactive:
        cmd.append("--interactive")
    
    # Add explainers if provided
    if args.explainers:
        # Split the comma-separated string into individual explainers
        explainers = args.explainers.split(',')
        cmd.append("--explainers")
        # Add each explainer as a separate argument
        for explainer in explainers:
            cmd.append(explainer.strip())
    
    # Add notification flags if requested
    if args.notify:
        cmd.append("--notify")
        cmd.extend(["--notify-threshold", str(args.notify_threshold)])
        
    # Drift type is used for data preparation only, not passed to main_v2.py
    
    # Run command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("MoE validation completed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"MoE validation failed with exit code {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main())
