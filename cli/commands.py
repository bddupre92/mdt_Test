import argparse
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import sys
from pathlib import Path
import traceback
from baseline_comparison.comparison_runner import BaselineComparison
from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
from baseline_comparison.benchmark_utils import get_benchmark_function, get_all_benchmark_functions
from baseline_comparison.visualization import ComparisonVisualizer
from meta_optimizer import MetaOptimizer
from meta_optimizer.optimizers import load_optimizers
from meta_optimizer.benchmark_functions import BENCHMARK_FUNCTIONS
import matplotlib.pyplot as plt
import numpy as np
import json
import datetime
import os
from collections import Counter

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

class MoEValidationCommand(Command):
    """Command for running MoE validation framework tests"""
    def execute(self) -> Dict[str, Any]:
        """Execute MoE validation framework tests"""
        from core.moe_validation import run_moe_validation
        
        self.logger.info("Running MoE validation framework tests")
        return run_moe_validation(self.args)

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
            # Get parameters from args
            dimensions = self.args.dimensions
            max_evaluations = self.args.max_evaluations
            num_trials = self.args.num_trials
            output_dir = self.args.output_dir
            model_path = self.args.selector_path
            
            # Parse functions to test
            if self.args.all_functions:
                logger.info(f"Using all benchmark functions")
                functions = ["sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "levy", "schwefel"]
            else:
                functions = self.args.functions
                if isinstance(functions, str):
                    functions = functions.split(",")
                logger.info(f"Using specified functions: {functions}")
            
            # Create timestamped output directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_dir, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")
            
            # Initialize all four optimizers
            
            # 1. Simple Baseline (Random Selection)
            from baseline_comparison.baseline_algorithms.simple_baseline import SimpleBaselineSelector
            simple_baseline = SimpleBaselineSelector()
            logger.info("Initialized SimpleBaselineSelector")
            
            # 2. Meta-Learner (Basic Bandit)
            meta_learner = MetaOptimizer(
                dimensions=dimensions,
                max_evaluations=max_evaluations,
                n_parallel=1,
                visualize_selection=True
            )
            logger.info("Initialized Meta-Learner")
            
            # 3. Enhanced Meta-Optimizer
            enhanced_meta = MetaOptimizer(
                dimensions=dimensions,
                max_evaluations=max_evaluations,
                n_parallel=1,
                visualize_selection=True,
                use_ml_selection=True,
                extract_features=True
            )
            logger.info("Initialized Enhanced Meta-Optimizer")
            
            # 4. SATzilla-Inspired Selector
            satzilla_selector = SatzillaInspiredSelector()
            logger.info("Initialized SatzillaInspiredSelector")
            
            # Load optimizer implementations for all selectors
            try:
                optimizers = load_optimizers()
                meta_learner.register_optimizers(optimizers)
                enhanced_meta.register_optimizers(optimizers)
                simple_baseline.set_available_algorithms(list(optimizers.keys()))
                satzilla_selector.set_available_algorithms(list(optimizers.keys()))
                logger.info(f"Loaded {len(optimizers)} optimizers: {list(optimizers.keys())}")
            except ImportError as e:
                logger.error(f"Could not import optimizer implementations: {e}")
                return 1
            
            # Load SATzilla model if provided
            if model_path:
                model_path_str = str(model_path)
                logger.info(f"Loading SATzilla model from: {model_path_str}")
                try:
                    satzilla_selector.load_model(model_path_str)
                    if satzilla_selector.is_trained:
                        logger.info("Successfully loaded SATzilla model")
                    else:
                        logger.warning("Model loaded but is_trained=False. The model may be invalid.")
                except Exception as e:
                    logger.error(f"Failed to load SATzilla model: {e}")
                    return 1
            
            # Initialize comparison framework
            comparison = BaselineComparison(
                simple_baseline=simple_baseline,
                meta_learner=meta_learner,
                enhanced_meta=enhanced_meta,
                satzilla_selector=satzilla_selector,
                max_evaluations=max_evaluations,
                num_trials=num_trials,
                verbose=True,
                output_dir=output_dir,
                model_path=model_path
            )
            
            # Run comparison for each benchmark function
            logger.info("Running baseline comparison...")
            for func_name in functions:
                try:
                    # Get benchmark function
                    func = get_benchmark_function(func_name, dimensions)
                    if func is None:
                        logger.warning(f"Skipping unknown function: {func_name}")
                        continue
                    
                    # Run comparison
                    logger.info(f"Running comparison for {func_name}")
                    comparison.run_comparison(
                        problem_name=func_name,
                        problem_func=func,
                        dimensions=dimensions,
                        max_evaluations=max_evaluations,
                        num_trials=num_trials
                    )
                except Exception as e:
                    logger.error(f"Error running comparison for {func_name}: {e}")
                    traceback.print_exc()
            
            # Generate visualizations
            logger.info("Generating visualizations...")
            visualizer = ComparisonVisualizer(comparison.results, export_dir=os.path.join(output_dir, "visualizations"))
            visualizer.create_all_visualizations()
            
            # Save summary information
            with open(os.path.join(output_dir, "comparison_summary.txt"), "w") as f:
                f.write(f"Baseline Comparison Summary\n")
                f.write(f"==========================\n")
                f.write(f"Dimensions: {dimensions}\n")
                f.write(f"Max Evaluations: {max_evaluations}\n")
                f.write(f"Number of Trials: {num_trials}\n")
                f.write(f"Model Path: {model_path if model_path else 'None'}\n")
                f.write(f"SATzilla Model Loaded: {satzilla_selector.is_trained}\n")
                f.write(f"Benchmark Functions: {functions}\n")
                f.write(f"Timestamp: {timestamp}\n")
            
            logger.info(f"Baseline comparison completed successfully. Results saved to: {output_dir}")
            return 0
            
        except Exception as e:
            logger.error(f"Error in baseline comparison: {str(e)}")
            traceback.print_exc()
            return 1

class SatzillaTrainCommand(Command):
    """Command for training a SATzilla-inspired algorithm selector"""
    
    def execute(self) -> int:
        """
        Execute the training of a SATzilla-inspired algorithm selector
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Get parameters from args
            dimensions = self.args.get('dimensions', 2)
            max_evaluations = self.args.get('max_evaluations', 1000)
            output_file = self.args.get('output_file')
            
            # Validate output_file
            if not output_file:
                print("Error: output_file is required")
                return 1
            
            # Get the list of benchmark functions
            if self.args.get('all_functions', False):
                # Use all available benchmark functions
                benchmark_functions = get_all_benchmark_functions(dimensions)
                print(f"Using all {len(benchmark_functions)} benchmark functions for training")
            else:
                # Use specified functions or defaults
                function_names = self.args.get('functions', ['sphere', 'rosenbrock'])
                benchmark_functions = [
                    get_benchmark_function(name, dimensions) 
                    for name in function_names
                ]
                print(f"Using specified benchmark functions for training: {[func.name for func in benchmark_functions]}")
            
            # Initialize the selector
            selector = SatzillaInspiredSelector()
            
            # Train the selector
            print("Training SATzilla-inspired selector...")
            selector.train(benchmark_functions, max_evaluations=max_evaluations)
            
            # Save the trained model
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure the file has .joblib extension for clarity
            if not str(output_path).endswith('.joblib'):
                output_path = Path(f"{str(output_path)}.joblib")
                
            selector.save_model(str(output_path))
            print(f"Trained selector saved to: {output_path}")
            
            return 0
            
        except Exception as e:
            print(f"Error training SATzilla-inspired selector: {e}")
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
    elif args.moe_validation:
        return MoEValidationCommand(args)
    elif args.import_migraine_data:
        return MigraineDataImportCommand(args)
    elif args.predict_migraine:
        return MigrainePredictionCommand(args)
    elif args.baseline_comparison:
        return BaselineComparisonCommand(args)
    elif args.train_satzilla:
        return SatzillaTrainCommand(args)
    else:
        return None
