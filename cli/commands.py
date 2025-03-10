import argparse
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class MetaLearningCommand(Command):
    """Command for running meta-learning"""
    def execute(self) -> Dict[str, Any]:
        """Execute meta-learning"""
        from core.meta_learning import run_meta_learning
        
        self.logger.info("Running meta-learning")
        return run_meta_learning(self.args)

class EnhancedMetaLearningCommand(Command):
    """Command for running enhanced meta-learning"""
    def execute(self) -> Dict[str, Any]:
        """Execute enhanced meta-learning"""
        from core.meta_learning import run_enhanced_meta_learning
        
        self.logger.info("Running enhanced meta-learning")
        return run_enhanced_meta_learning(self.args)

class OptimizationCommand(Command):
    """Command for running optimization"""
    def execute(self) -> Dict[str, Any]:
        """Execute optimization"""
        from core.optimization import run_optimization
        
        self.logger.info(f"Running optimization with {self.args.optimizer}")
        return run_optimization(self.args)

class OptimizerComparisonCommand(Command):
    """Command for comparing optimizers"""
    def execute(self) -> Dict[str, Any]:
        """Execute optimizer comparison"""
        from core.optimization import run_optimizer_comparison
        
        self.logger.info("Running optimizer comparison")
        return run_optimizer_comparison(self.args)

class EvaluationCommand(Command):
    """Command for running model evaluation"""
    def execute(self) -> Dict[str, Any]:
        """Execute model evaluation"""
        from core.evaluation import run_evaluation
        
        self.logger.info("Running model evaluation")
        return run_evaluation(self.args)

class DriftDetectionCommand(Command):
    """Command for running drift detection"""
    def execute(self) -> Dict[str, Any]:
        """Execute drift detection"""
        from core.drift_detection import run_drift_detection
        
        self.logger.info("Running drift detection")
        return run_drift_detection(self.args)

class ExplainabilityCommand(Command):
    """Command for running explainability analysis"""
    def execute(self) -> Dict[str, Any]:
        """Execute explainability analysis"""
        from explainability.model_explainer import run_explainability_analysis
        
        self.logger.info(f"Running explainability analysis with {self.args.explainer}")
        return run_explainability_analysis(self.args)

class DriftExplainerCommand(Command):
    """Command for explaining drift"""
    def execute(self) -> Dict[str, Any]:
        """Execute drift explanation"""
        from explainability.drift_explainer import explain_drift
        
        self.logger.info("Running drift explanation")
        return explain_drift(self.args)

class MigraineDataImportCommand(Command):
    """Command for importing migraine data"""
    def execute(self) -> Dict[str, Any]:
        """Execute migraine data import"""
        from migraine.data_import import run_migraine_data_import
        
        self.logger.info("Importing migraine data")
        return run_migraine_data_import(self.args)

class MigrainePredictionCommand(Command):
    """Command for running migraine prediction"""
    def execute(self) -> Dict[str, Any]:
        """Execute migraine prediction"""
        from migraine.prediction import run_migraine_prediction
        
        self.logger.info("Running migraine prediction")
        return run_migraine_prediction(self.args)

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
                logger.info("Using real MetaOptimizer")
            except ImportError:
                logger.warning("Could not import MetaOptimizer. Using mock instead.")
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
                logger.info(f"Using all {len(benchmark_functions)} benchmark functions")
            else:
                # Use specified functions or defaults
                function_names = self.args.get('functions', ['sphere', 'rosenbrock'])
                benchmark_functions = [
                    get_benchmark_function(name, dimensions) 
                    for name in function_names
                ]
                logger.info(f"Using specified benchmark functions: {[func.name for func in benchmark_functions]}")
            
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
            logger.info("Running baseline comparison...")
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
                logger.info("Generating visualizations...")
                
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
                    logger.warning("ComparisonVisualizer not available. Using basic visualizations only.")
            
            # Save results as JSON
            logger.info("Saving results...")
            
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
            
            logger.info("Baseline comparison completed successfully!")
            return 0
            
        except Exception as e:
            logger.error(f"Error in baseline comparison: {e}")
            import traceback
            traceback.print_exc()
            return 1

def get_command(args: argparse.Namespace) -> Optional[Command]:
    """
    Get the appropriate command based on parsed arguments
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command-line arguments
    
    Returns:
    --------
    Optional[Command]
        Command to execute, or None if no command is specified
    """
    if args.meta:
        return MetaLearningCommand(args)
    elif args.enhanced_meta:
        return EnhancedMetaLearningCommand(args)
    elif args.optimize:
        return OptimizationCommand(args)
    elif args.compare_optimizers:
        return OptimizerComparisonCommand(args)
    elif args.evaluate:
        return EvaluationCommand(args)
    elif args.drift:
        return DriftDetectionCommand(args)
    elif args.explain or args.auto_explain:
        return ExplainabilityCommand(args)
    elif args.explain_drift:
        return DriftExplainerCommand(args)
    elif args.import_migraine_data:
        return MigraineDataImportCommand(args)
    elif args.predict_migraine:
        return MigrainePredictionCommand(args)
    elif args.baseline_comparison:
        return BaselineComparisonCommand(args)
    else:
        return None
