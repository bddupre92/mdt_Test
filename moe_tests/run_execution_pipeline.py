#!/usr/bin/env python
"""
Run the MoE Execution Pipeline

This script demonstrates how to use the ExecutionPipeline to run a complete
MoE workflow with EC algorithms.
"""

import os
import sys
import argparse
import logging
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import required modules
from moe_framework.execution.execution_pipeline import ExecutionPipeline
from moe_tests.conftest import sample_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("execution_pipeline")

def main():
    """Run the MoE Execution Pipeline."""
    parser = argparse.ArgumentParser(description='Run the MoE Execution Pipeline')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data file (CSV). If not provided, sample data will be generated.')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or generate data
    if args.data and os.path.exists(args.data):
        logger.info(f"Loading data from {args.data}")
        data_path = args.data
    else:
        logger.info("Generating sample data")
        data = sample_data()
        
        # Save sample data for reference
        data_path = os.path.join(args.output_dir, 'sample_data.csv')
        data.to_csv(data_path, index=False)
        logger.info(f"Saved sample data to {data_path}")
    
    # Define target column
    target_column = 'migraine_severity'
    
    # Create configuration
    config = {
        'output_dir': args.output_dir,
        'environment': 'dev',
        'upload': {
            'allowed_extensions': ['csv', 'xlsx'],
            'max_file_size_mb': 10
        }
    }
    
    # Initialize execution pipeline
    execution_pipeline = ExecutionPipeline(config=config, verbose=True)
    
    # Execute the pipeline
    result = execution_pipeline.execute(
        data_path=data_path,
        target_column=target_column
    )
    
    if result['success']:
        logger.info(f"Execution completed successfully. Results saved to {result['results_path']}")
    else:
        logger.error(f"Execution failed: {result['message']}")

if __name__ == "__main__":
    main() 