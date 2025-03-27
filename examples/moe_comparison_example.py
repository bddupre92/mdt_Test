#!/usr/bin/env python3
"""
Example script demonstrating how to use the MoE comparison functionality.

This script provides a simple example of using the MoE comparison command
through both programmatic API and command-line interface.
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path to allow imports when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import directly from the commands.py file, not the commands/ directory
import sys
import importlib.util
import os

# Get the absolute path to commands.py
commands_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cli', 'commands.py')

# Load the module directly
spec = importlib.util.spec_from_file_location("commands_module", commands_path)
commands_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(commands_module)

# Now we can access MoEComparisonCommand directly
MoEComparisonCommand = commands_module.MoEComparisonCommand

from baseline_comparison.moe_comparison import create_moe_adapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_default_config():
    """
    Create a default MoE configuration for testing.
    
    Returns:
        dict: Default MoE configuration
    """
    return {
        "moe_framework": {
            "gating_network": {
                "type": "confidence_based",
                "hidden_layers": [64, 32],
                "activation": "relu",
                "dropout_rate": 0.2
            },
            "experts": {
                "count": 5,
                "specialization": "function_based",
                "types": ["global", "local", "hybrid", "exploratory", "exploitative"]
            },
            "integration": {
                "method": "weighted_average",
                "confidence_threshold": 0.6,
                "min_experts": 2
            },
            "training": {
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.001,
                "early_stopping_patience": 10
            }
        }
    }

def save_default_config(output_path):
    """
    Save the default MoE configuration to a file.
    
    Args:
        output_path (str): Path to save the configuration
    
    Returns:
        str: Path to the saved configuration file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(create_default_config(), f, indent=2)
    logger.info(f"Saved default MoE configuration to {output_path}")
    return output_path

def run_moe_comparison_programmatically():
    """Run MoE comparison using the programmatic API."""
    # Create a temporary config file
    config_path = save_default_config("./examples/temp_moe_config.json")
    
    # Create command arguments
    args = {
        "moe_comparison": True,
        "moe_config_path": config_path,
        "moe_model_path": None,
        "output_dir": "results/moe_comparison_example",
        "num_trials": 5,
        "functions": ["sphere", "rosenbrock", "rastrigin"],
        "all_functions": False,
        "visualize_moe_contributions": True,
        "calculate_expert_impact": True,
        "detailed_report": True,
        "include_confidence_metrics": True,
    }
    
    # Create and execute the command
    command = MoEComparisonCommand(args)
    result = command.execute()
    
    logger.info(f"MoE comparison completed with result code: {result}")
    logger.info(f"Results saved to: {args['output_dir']}")

def main():
    """Main function for the example script."""
    parser = argparse.ArgumentParser(description="MoE Comparison Example")
    parser.add_argument('--mode', choices=['api', 'cli'], default='api',
                      help="Run mode: use Python API or show CLI command")
    args = parser.parse_args()
    
    if args.mode == 'api':
        # Run using the programmatic API
        run_moe_comparison_programmatically()
    else:
        # Show how to run via CLI
        config_path = save_default_config("./examples/temp_moe_config.json")
        cli_command = (
            "python -m cli.main moe_comparison "
            f"--moe-config-path {config_path} "
            "--output-dir results/moe_comparison_example "
            "--num-trials 5 "
            "--functions sphere rosenbrock rastrigin "
            "--visualize-moe-contributions "
            "--calculate-expert-impact "
            "--detailed-report "
            "--include-confidence-metrics"
        )
        print("\nTo run the MoE comparison via CLI, use the following command:")
        print(f"\n{cli_command}\n")

if __name__ == "__main__":
    main()
