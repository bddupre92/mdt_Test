#!/usr/bin/env python
"""
Run an example MoE workflow with tracking enabled.

This script demonstrates how to run a complete MoE workflow with tracking
enabled, which will generate data for the dashboard visualization.
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
from moe_framework.workflow.moe_pipeline import MoEPipeline
from moe_framework.event_tracking.workflow_tracker import WorkflowTracker
from moe_tests.conftest import sample_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("example_workflow")

def main():
    """Run an example MoE workflow with tracking enabled."""
    parser = argparse.ArgumentParser(description='Run an example MoE workflow')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data file (CSV). If not provided, sample data will be generated.')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--tracking-dir', type=str, default='./.workflow_tracking',
                        help='Directory to save workflow tracking data')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tracking_dir, exist_ok=True)
    
    # Load or generate data
    if args.data and os.path.exists(args.data):
        logger.info(f"Loading data from {args.data}")
        data = pd.read_csv(args.data)
    else:
        logger.info("Generating sample data")
        data = sample_data()
        
        # Save sample data for reference
        sample_data_path = os.path.join(args.output_dir, 'sample_data.csv')
        data.to_csv(sample_data_path, index=False)
        logger.info(f"Saved sample data to {sample_data_path}")
    
    # Define target column
    target_column = 'migraine_severity'
    
    # Create configuration
    config = {
        'output_dir': args.output_dir,
        'environment': 'dev',
        'execution': {
            'output_dir': args.output_dir
        },
        'experts': {
            'physiological': {
                'enabled': True,
                'optimize_hyperparams': True
            },
            'behavioral': {
                'enabled': True,
                'optimize_hyperparams': True
            },
            'environmental': {
                'enabled': True,
                'optimize_hyperparams': True
            },
            'medication_history': {
                'enabled': True,
                'optimize_hyperparams': True
            }
        },
        'gating': {
            'type': 'meta_learner',
            'optimize_weights': True
        },
        'meta_learner': {
            'enabled': True
        }
    }
    
    # Initialize workflow tracker
    tracker = WorkflowTracker(output_dir=args.tracking_dir, verbose=True)
    
    # Initialize pipeline
    pipeline = MoEPipeline(config=config, verbose=True)
    
    # Track the pipeline
    tracked_pipeline = tracker.track_moe_pipeline(pipeline)
    
    # Start workflow tracking
    workflow_id = tracker.start_workflow("example_workflow")
    logger.info(f"Started workflow tracking with ID: {workflow_id}")
    
    try:
        # Load data
        tracked_pipeline.load_data(data, target_column=target_column)
        
        # Train the pipeline
        tracked_pipeline.train(validation_split=0.2, random_state=42)
        
        # Make predictions
        test_data = data.sample(frac=0.2, random_state=42)
        predictions = tracked_pipeline.predict(test_data)
        
        # Evaluate the pipeline
        evaluation = tracked_pipeline.evaluate(test_data=test_data, test_target=test_data[target_column])
        
        # Complete workflow tracking
        tracker.complete_workflow(
            success=True,
            results={
                'evaluation': evaluation,
                'config': config
            }
        )
        
        logger.info(f"Workflow completed successfully. Results saved to {args.output_dir}")
        logger.info(f"Workflow tracking data saved to {args.tracking_dir}")
        logger.info("You can now run the dashboard to visualize the workflow execution.")
        
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}")
        tracker.complete_workflow(
            success=False,
            results={
                'error': str(e)
            }
        )
        raise

if __name__ == "__main__":
    main() 