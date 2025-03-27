#!/usr/bin/env python3
"""
Generate MOE Publication Report

This script generates a publication-ready report for the MOE framework,
including data preprocessing, synthetic data generation, and visualizations.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

from app.reporting.unified_report_generator import UnifiedReportGenerator
from app.reporting.modules.moe_publication_report import (
    preprocess_data,
    generate_synthetic_data,
    generate_publication_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready MOE framework report"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input data file (CSV)"
    )
    
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of the target column"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/moe_publication",
        help="Output directory for results and report"
    )
    
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to existing MOE checkpoint file"
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess real data
    logger.info("Loading and preprocessing data...")
    real_data = pd.read_csv(args.data)
    X_scaled, y, scaler = preprocess_data(real_data, args.target)
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    synthetic_data = generate_synthetic_data(real_data, args.synthetic_samples)
    
    # Load or generate MOE results
    if args.checkpoint:
        logger.info(f"Loading MOE results from checkpoint: {args.checkpoint}")
        with open(args.checkpoint, 'r') as f:
            test_results = json.load(f)
    else:
        logger.info("No checkpoint provided. Running MOE pipeline...")
        from run_moe_pipeline import main as run_moe_pipeline
        
        # Set up sys.argv for the MOE pipeline
        sys.argv = [
            'run_moe_pipeline.py',
            '--data', args.data,
            '--target', args.target,
            '--output', output_dir,
            '--visualize'
        ]
        
        # Run the pipeline
        run_moe_pipeline()
        
        # Load the latest checkpoint from dev/checkpoints subdirectory
        checkpoint_dir = Path(output_dir) / 'dev' / 'checkpoints'
        if not checkpoint_dir.exists():
            raise RuntimeError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        checkpoints = sorted(checkpoint_dir.glob('*.json'), key=lambda x: x.stat().st_mtime)
        if not checkpoints:
            raise RuntimeError(f"No checkpoints found in {checkpoint_dir}")
            
        logger.info(f"Loading checkpoint: {checkpoints[-1]}")
        with open(checkpoints[-1], 'r') as f:
            test_results = json.load(f)
    
    # Add additional information to results
    test_results.update({
        'output_dir': output_dir,
        'target_column': args.target,
        'timestamp': timestamp,
        'data_info': {
            'n_samples': len(real_data),
            'n_features': len(real_data.columns) - 1,
            'feature_names': list(real_data.columns),
            'synthetic_samples': args.synthetic_samples
        }
    })
    
    # Generate publication report
    logger.info("Generating publication report...")
    report_path = generate_publication_report(
        test_results=test_results,
        real_data=real_data,
        synthetic_data=synthetic_data
    )
    
    logger.info(f"Report generated successfully: {report_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
