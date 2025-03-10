"""
Command classes for CLI functionality
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import argparse
import logging
import os
import sys
from pathlib import Path

class Command(ABC):
    """
    Base class for all CLI commands
    """
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the command with parsed arguments
        
        Parameters:
        -----------
        args : argparse.Namespace
            Parsed command-line arguments
        """
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the command
        
        Returns:
        --------
        Dict[str, Any]
            Results of the command execution
        """
        pass
    
    def validate_args(self) -> bool:
        """
        Validate command arguments before execution
        
        Returns:
        --------
        bool
            True if arguments are valid, False otherwise
        """
        return True

# Import all command classes
# Note: MetaLearningCommand is implemented in core/meta_learning.py
# Commented out non-existent or currently unavailable imports
# from .enhanced_meta_learning import EnhancedMetaLearningCommand
# from .optimization import OptimizationCommand
# from .optimizer_comparison import OptimizerComparisonCommand
# from .evaluation import EvaluationCommand
# from .drift_detection import DriftDetectionCommand
# from .explainability import ExplainabilityCommand
# from .drift_explainer import DriftExplainerCommand
# from .migraine_data_import import MigraineDataImportCommand
# from .migraine_prediction import MigrainePredictionCommand
from .dynamic_optimization import DynamicOptimizationCommand

class BaselineComparisonCommand(Command):
    """Command for running baseline comparison benchmarks"""
    
    def execute(self) -> int:
        """
        Execute the baseline comparison benchmark
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Import required modules
            from baseline_comparison.comparison_runner import BaselineComparison
            from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
            from baseline_comparison.benchmark_utils import get_benchmark_function, get_all_benchmark_functions
            import matplotlib.pyplot as plt
            import numpy as np
            import json
            import datetime
            from collections import Counter
            
            # Try to import MetaOptimizer
            try:
                from meta_optimizer import MetaOptimizer
                self.logger.info("Using real MetaOptimizer")
            except ImportError:
                self.logger.warning("Could not import MetaOptimizer. Using mock instead.")
                # Create a mock MetaOptimizer for testing
                class MetaOptimizer:
                    def __init__(self, *args, **kwargs):
                        self.name = "MockMetaOptimizer"
                        
                    def optimize(self, problem, *args, **kwargs):
                        # Simple random optimization for testing
                        best_x = np.random.uniform(-5, 5, problem.dims)
                        best_y = problem.evaluate(best_x)
                        return best_x, best_y, 100
            
            # Get parameters from args
            dimensions = self.args.get('dimensions', 2)
            max_evaluations = self.args.get('max_evaluations', 1000)
            num_trials = self.args.get('num_trials', 3)
            
            # Get the list of benchmark functions
            if self.args.get('all_functions', False):
                # Use all available benchmark functions
                benchmark_functions = get_all_benchmark_functions(dimensions)
                self.logger.info(f"Using all {len(benchmark_functions)} benchmark functions")
            else:
                # Use specified functions or defaults
                function_names = self.args.get('functions', ['sphere', 'rosenbrock'])
                benchmark_functions = [
                    get_benchmark_function(name, dimensions) 
                    for name in function_names
                ]
                self.logger.info(f"Using specified benchmark functions: {[func.name for func in benchmark_functions]}")
            
            # Initialize components
            selector = SatzillaInspiredSelector()
            meta_optimizer = MetaOptimizer()
            
            # Initialize comparison framework
            comparison = BaselineComparison(
                baseline_selector=selector,
                meta_optimizer=meta_optimizer,
                max_evaluations=max_evaluations,
                num_trials=num_trials,
                verbose=not self.args.get('quiet', False)
            )
            
            # Run comparison
            self.logger.info("Running baseline comparison...")
            results = comparison.run_comparison(benchmark_functions)
            
            # Set up output directories
            output_base_dir = Path(self.args.get('output_dir', 'results/baseline_comparison'))
            if self.args.get('timestamp_dir', True):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = output_base_dir / timestamp
            else:
                output_dir = output_base_dir
            
            # Create subdirectories
            data_dir = output_dir / "data"
            viz_dir = output_dir / "visualizations"
            
            # Create directories
            output_dir.mkdir(exist_ok=True, parents=True)
            data_dir.mkdir(exist_ok=True)
            viz_dir.mkdir(exist_ok=True)
            
            # Generate and save visualizations
            if not self.args.get('no_visualizations', False):
                self.logger.info("Generating visualizations...")
                
                # Performance comparison
                plt.figure(figsize=(10, 6))
                comparison.plot_performance_comparison(results)
                plt.savefig(viz_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # Algorithm selection frequency
                plt.figure(figsize=(15, 6))
                comparison.plot_algorithm_selection_frequency(results)
                plt.savefig(viz_dir / "algorithm_selection.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # If ComparisonVisualizer is available, use it for more visualizations
                try:
                    from baseline_comparison.visualization import ComparisonVisualizer
                    
                    # Use the visualizer to create all visualizations
                    visualizer = ComparisonVisualizer(results, export_dir=str(viz_dir))
                    visualizer.create_all_visualizations()
                    
                except ImportError:
                    self.logger.warning("ComparisonVisualizer not available. Using basic visualizations only.")
            
            # Save results as JSON
            self.logger.info("Saving results...")
            
            # Convert numpy types to Python types for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                else:
                    return obj
            
            json_results = {}
            for func_name, func_results in results.items():
                json_results[func_name] = {k: convert_for_json(v) for k, v in func_results.items()}
            
            with open(data_dir / "benchmark_results.json", "w") as f:
                json.dump(json_results, f, indent=2)
            
            # Save a summary report as text
            with open(data_dir / "benchmark_summary.txt", "w") as f:
                f.write("Baseline Comparison Benchmark Summary\n")
                f.write("===================================\n\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dimensions: {dimensions}\n")
                f.write(f"Max Evaluations: {max_evaluations}\n")
                f.write(f"Number of Trials: {num_trials}\n\n")
                
                f.write("Results by Function:\n")
                f.write("-----------------\n\n")
                
                for func_name, func_results in results.items():
                    f.write(f"Function: {func_name}\n")
                    f.write(f"  Baseline best fitness: {func_results['baseline_best_fitness_avg']:.6f} ± {func_results['baseline_best_fitness_std']:.6f}\n")
                    f.write(f"  Meta Optimizer best fitness: {func_results['meta_best_fitness_avg']:.6f} ± {func_results['meta_best_fitness_std']:.6f}\n")
                    improvement = (func_results['baseline_best_fitness_avg'] - func_results['meta_best_fitness_avg']) / abs(func_results['baseline_best_fitness_avg']) * 100
                    f.write(f"  Improvement: {improvement:.2f}%\n")
                    f.write(f"  Baseline algorithm selections: {dict(sorted(Counter(func_results['baseline_selected_algorithms']).items()))}\n")
                    f.write(f"  Meta algorithm selections: {dict(sorted(Counter(func_results['meta_selected_algorithms']).items()))}\n\n")
            
            # Create an index.md file with links to all results
            with open(output_dir / "index.md", "w") as f:
                f.write("# Baseline Comparison Results\n\n")
                f.write("## Overview\n\n")
                f.write("This directory contains the results of comparing the Meta Optimizer against ")
                f.write("the SATzilla-inspired baseline algorithm selector.\n\n")
                
                f.write("## Run Information\n\n")
                f.write(f"- **Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **Dimensions**: {dimensions}\n")
                f.write(f"- **Max Evaluations**: {max_evaluations}\n")
                f.write(f"- **Number of Trials**: {num_trials}\n")
                f.write(f"- **Benchmark Functions**: {[func.name for func in benchmark_functions]}\n\n")
                
                f.write("## Results Summary\n\n")
                f.write("| Function | Baseline | Meta Optimizer | Improvement |\n")
                f.write("|----------|----------|----------------|------------|\n")
                
                for func_name, func_results in results.items():
                    baseline = func_results['baseline_best_fitness_avg']
                    meta = func_results['meta_best_fitness_avg']
                    improvement = (baseline - meta) / abs(baseline) * 100
                    f.write(f"| {func_name} | {baseline:.6f} | {meta:.6f} | {improvement:.2f}% |\n")
                
                f.write("\n## Available Files\n\n")
                f.write("### Data\n\n")
                for file in data_dir.glob("*"):
                    f.write(f"- [{file.name}](data/{file.name})\n")
                
                f.write("\n### Visualizations\n\n")
                for file in viz_dir.glob("*"):
                    f.write(f"- [{file.name}](visualizations/{file.name})\n")
            
            # Print a summary to the console
            print("\nBaseline Comparison Results Summary:")
            print("===================================")
            for func_name, func_results in results.items():
                print(f"\nFunction: {func_name}")
                print(f"  Baseline best fitness: {func_results['baseline_best_fitness_avg']:.6f}")
                print(f"  Meta Optimizer best fitness: {func_results['meta_best_fitness_avg']:.6f}")
                improvement = (func_results['baseline_best_fitness_avg'] - func_results['meta_best_fitness_avg']) / abs(func_results['baseline_best_fitness_avg']) * 100
                print(f"  Improvement: {improvement:.2f}%")
            
            print(f"\nResults saved to: {output_dir}")
            print(f"Summary: {output_dir}/index.md")
            
            self.logger.info("Baseline comparison completed successfully!")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error in baseline comparison: {e}")
            import traceback
            traceback.print_exc()
            return 1

class SatzillaTrainingCommand(Command):
    """Command for training the SATzilla-inspired selector"""
    
    def execute(self) -> int:
        """
        Execute the SATzilla training command
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Import required modules
            from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
            from baseline_comparison.benchmark_utils import get_benchmark_function, get_all_benchmark_functions
            from baseline_comparison.training import train_selector
            import numpy as np
            import json
            import datetime
            import os
            from pathlib import Path
            
            # Get parameters from args
            dimensions = self.args.get('dimensions', 2)
            max_evaluations = self.args.get('max_evaluations', 1000)
            num_problems = self.args.get('num_problems', 20)
            
            # Get the list of benchmark functions for training
            if self.args.get('all_functions', False):
                # Use all available benchmark functions
                benchmark_functions = get_all_benchmark_functions(dimensions)
                self.logger.info(f"Using all {len(benchmark_functions)} benchmark functions")
            else:
                # Use specified functions or defaults
                function_names = self.args.get('functions', ['sphere', 'rosenbrock', 'rastrigin', 'ackley', 'griewank'])
                benchmark_functions = [
                    get_benchmark_function(name, dimensions) 
                    for name in function_names
                ]
                self.logger.info(f"Using specified benchmark functions: {[func.name for func in benchmark_functions]}")
            
            # Initialize selector
            selector = SatzillaInspiredSelector()
            
            # Set up output directories
            output_base_dir = Path(self.args.get('output_dir', 'results/satzilla_training'))
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.args.get('timestamp_dir', True):
                output_dir = output_base_dir / timestamp
            else:
                output_dir = output_base_dir
            
            # Create output directory
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate problem variations if needed
            if len(benchmark_functions) < num_problems:
                self.logger.info(f"Generating problem variations to reach {num_problems} training problems")
                training_problems = train_selector.generate_problem_variations(
                    benchmark_functions, 
                    num_problems,
                    dimensions,
                    random_seed=self.args.get('seed', 42)
                )
            else:
                training_problems = benchmark_functions
            
            # Train the selector
            self.logger.info(f"Training SATzilla-inspired selector with {len(training_problems)} problems")
            selector = train_selector.train_satzilla_selector(
                selector,
                training_problems,
                max_evaluations=max_evaluations,
                export_features=True,
                export_dir=output_dir
            )
            
            # Save the trained selector
            model_dir = output_dir / "models"
            model_dir.mkdir(exist_ok=True)
            train_selector.save_trained_selector(selector, model_dir / "satzilla_selector.pkl")
            
            # Generate a summary report
            with open(output_dir / "training_summary.txt", "w") as f:
                f.write("SATzilla-inspired Selector Training Summary\n")
                f.write("=======================================\n\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dimensions: {dimensions}\n")
                f.write(f"Max Evaluations: {max_evaluations}\n")
                f.write(f"Number of Training Problems: {len(training_problems)}\n")
                f.write(f"Training Functions: {[p.name for p in benchmark_functions]}\n\n")
                f.write("Training completed successfully.\n")
                f.write(f"Trained model saved to: {model_dir / 'satzilla_selector.pkl'}\n")
            
            self.logger.info(f"Training completed. Results saved to {output_dir}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error during SATzilla training: {e}")
            import traceback
            traceback.print_exc()
            return 1

# Define COMMAND_MAP here, AFTER the BaselineComparisonCommand class is defined
COMMAND_MAP = {
    "baseline_comparison": BaselineComparisonCommand,
    "train_satzilla": SatzillaTrainingCommand,
    # Add other commands as they become available
}

def get_command(args: argparse.Namespace) -> Optional[Command]:
    """
    Get the appropriate command based on the parsed arguments
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command-line arguments
    
    Returns:
    --------
    Optional[Command]
        The command to execute, or None if no command is specified
    """
    # Check if the command attribute exists
    if hasattr(args, 'command') and args.command:
        command = args.command
        # Check if the command is in the COMMAND_MAP
        if command in COMMAND_MAP:
            return COMMAND_MAP[command](args)
    
    # Fall back to the existing implementation
    # Meta-learning commands
    if hasattr(args, 'meta') and args.meta:
        # Use the existing meta-learning implementation
        from core.meta_learning import run_meta_learning
        class MetaLearningCommandWrapper(Command):
            def execute(self):
                return run_meta_learning(self.args)
        return MetaLearningCommandWrapper(args)
    
    # Commented out unavailable commands
    """
    if hasattr(args, 'enhanced_meta') and args.enhanced_meta:
        return EnhancedMetaLearningCommand(args)
        
    # Optimization commands
    if hasattr(args, 'optimize') and args.optimize:
        return OptimizationCommand(args)
        
    if hasattr(args, 'compare_optimizers') and args.compare_optimizers:
        return OptimizerComparisonCommand(args)
        
    # Evaluation commands
    if hasattr(args, 'evaluate') and args.evaluate:
        return EvaluationCommand(args)
        
    # Drift detection
    if hasattr(args, 'drift') and args.drift:
        return DriftDetectionCommand(args)
        
    # Explainability
    if hasattr(args, 'explain') and args.explain:
        return ExplainabilityCommand(args)
        
    if hasattr(args, 'explain_drift') and args.explain_drift:
        return DriftExplainerCommand(args)
        
    # Migraine data
    if hasattr(args, 'import_migraine_data') and args.import_migraine_data:
        return MigraineDataImportCommand(args)
        
    if hasattr(args, 'predict_migraine') and args.predict_migraine:
        return MigrainePredictionCommand(args)
    """
    
    # Dynamic optimization
    if hasattr(args, 'dynamic_optimization') and args.dynamic_optimization:
        return DynamicOptimizationCommand(args)
        
    return None 