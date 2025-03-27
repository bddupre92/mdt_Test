#!/usr/bin/env python
"""
Run the MoE Framework Dashboard

This script launches the interactive dashboard for visualizing MoE framework
workflow executions, optimizer performance, and expert contributions.
"""

import os
import sys
import argparse
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the dashboard module
from moe_framework.event_tracking.dashboard import render_workflow_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dashboard_runner")

def main():
    """Run the MoE Framework Dashboard."""
    parser = argparse.ArgumentParser(description='Run the MoE Framework Dashboard')
    parser.add_argument('--tracking-dir', type=str, default='./.workflow_tracking',
                        help='Directory containing workflow tracking data')
    parser.add_argument('--port', type=int, default=8501,
                        help='Port to run the dashboard on')
    
    args = parser.parse_args()
    
    logger.info(f"Starting MoE Framework Dashboard with tracking data from: {args.tracking_dir}")
    logger.info(f"Dashboard will be available at http://localhost:{args.port}")
    
    # Run the dashboard
    render_workflow_dashboard(args.tracking_dir)

if __name__ == "__main__":
    main()
