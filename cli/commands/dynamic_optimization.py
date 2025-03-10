"""
Command implementation for dynamic optimization visualization
"""

import logging
import argparse
from typing import Dict, Any

from cli.commands import Command

class DynamicOptimizationCommand(Command):
    """Command for running dynamic optimization visualization"""
    
    def execute(self) -> Dict[str, Any]:
        """Execute dynamic optimization visualization"""
        from visualization.dynamic_optimization_viz import run_dynamic_optimization_experiment
        
        # Extract parameters from arguments
        function_name = self.args.function
        drift_type = self.args.drift_type
        dim = self.args.dimension
        bounds = None  # Use default bounds
        drift_rate = self.args.drift_rate
        drift_interval = self.args.drift_interval
        severity = self.args.severity
        max_iterations = self.args.max_iterations
        reoptimize_interval = self.args.reoptimize_interval
        save_dir = self.args.export_dir or "results/visualizations"
        show_plot = self.args.show_plot
        
        self.logger.info(f"Running dynamic optimization visualization for {function_name} with {drift_type} drift")
        self.logger.info(f"Parameters: dim={dim}, drift_rate={drift_rate}, drift_interval={drift_interval}, "
                        f"severity={severity}, max_iterations={max_iterations}, reoptimize_interval={reoptimize_interval}")
        
        # Run the experiment
        results = run_dynamic_optimization_experiment(
            function_name=function_name,
            drift_type=drift_type,
            dim=dim,
            bounds=bounds,
            drift_rate=drift_rate,
            drift_interval=drift_interval,
            severity=severity,
            max_iterations=max_iterations,
            reoptimize_interval=reoptimize_interval,
            save_dir=save_dir,
            show_plot=show_plot
        )
        
        if results.get("success", False):
            viz_path = results.get("visualization_path", "")
            self.logger.info(f"Dynamic optimization visualization saved to {viz_path}")
            print(f"\nDynamic optimization visualization saved to {viz_path}")
            
            # Print drift characteristics
            drift_chars = results.get("drift_characteristics", {})
            if drift_chars:
                print("\nDrift Characteristics:")
                for key, value in drift_chars.items():
                    print(f"  {key}: {value}")
        else:
            error = results.get("error", "Unknown error")
            self.logger.error(f"Dynamic optimization visualization failed: {error}")
            print(f"Error: {error}")
        
        return results
    
    def validate_args(self) -> bool:
        """Validate command arguments before execution"""
        valid = True
        
        # Check if function name is provided
        if not hasattr(self.args, 'function') or not self.args.function:
            self.logger.error("Function name is required")
            valid = False
        
        # Check if drift type is provided
        if not hasattr(self.args, 'drift_type') or not self.args.drift_type:
            self.logger.error("Drift type is required")
            valid = False
            
        # Check if drift type is valid
        valid_drift_types = ['sudden', 'oscillatory', 'linear', 'incremental', 'gradual', 'random', 'noise']
        if hasattr(self.args, 'drift_type') and self.args.drift_type not in valid_drift_types:
            self.logger.error(f"Invalid drift type: {self.args.drift_type}. Valid types are: {', '.join(valid_drift_types)}")
            valid = False
        
        return valid 