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
            
            # Validate model path if provided
            if model_path:
                model_path = Path(model_path)
                if not model_path.exists():
                    # Try adding .joblib extension if not already present
                    if not str(model_path).endswith('.joblib'):
                        joblib_path = Path(f"{str(model_path)}.joblib")
                        if joblib_path.exists():
                            model_path = joblib_path
                            logger.info(f"Using model path with .joblib extension: {model_path}")
                        else:
                            # Try with .pkl extension as well
                            pkl_path = Path(f"{str(model_path)}.pkl")
                            if pkl_path.exists():
                                model_path = pkl_path
                                logger.info(f"Using model path with .pkl extension: {model_path}")
                            else:
                                logger.warning(f"Model file not found at: {model_path}, {joblib_path}, or {pkl_path}")
                    else:
                        logger.warning(f"Model file not found at: {model_path}")
            
            # Create timestamped output directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_dir, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")
            
            # Initialize meta optimizer
            meta_optimizer = MetaOptimizer(
                dimensions=dimensions,
                max_evaluations=max_evaluations,
                n_parallel=2,
                visualize_selection=True
            )
            
            # Load optimizer implementations
            try:
                optimizers = load_optimizers()
                meta_optimizer.register_optimizers(optimizers)
            except ImportError:
                logger.warning("Could not import optimizer implementations, using empty optimizer dictionary")
            
            # Get benchmark functions
            if self.args.all_functions:
                logger.info(f"Using all {len(BENCHMARK_FUNCTIONS)} benchmark functions")
                benchmark_functions = BENCHMARK_FUNCTIONS
            else:
                benchmark_functions = [BENCHMARK_FUNCTIONS[0]]  # Just use first function
            
            # Initialize baseline selector
            baseline_selector = SatzillaInspiredSelector()
            logger.info(f"Initialized SatzillaInspiredSelector")
            
            # Convert model_path to string if it's a Path object
            model_path_str = str(model_path) if model_path else None
            
            # Log the model path for debugging
            if model_path_str:
                logger.info(f"Using model file: {model_path_str}")
                
                # Check if file exists
                if os.path.exists(model_path_str):
                    logger.info(f"Verified model file exists: {model_path_str}")
                    
                    # Load model directly into the selector first
                    try:
                        logger.info(f"Loading model directly into selector: {model_path_str}")
                        baseline_selector.load_model(model_path_str)
                        if baseline_selector.is_trained:
                            logger.info(f"Successfully loaded model, is_trained={baseline_selector.is_trained}")
                        else:
                            logger.warning(f"Model loaded but is_trained=False. The model may be invalid.")
                    except Exception as e:
                        logger.error(f"Failed to load model directly into selector: {e}")
                        
                    # Try to load and check the model directly 
                    try:
                        import joblib
                        logger.info(f"Directly loading model to inspect it: {model_path_str}")
                        model_data = joblib.load(model_path_str)
                        
                        if isinstance(model_data, dict):
                            logger.info(f"Model data is a dictionary with keys: {list(model_data.keys())}")
                            if 'is_trained' in model_data:
                                logger.info(f"Model has is_trained flag set to: {model_data['is_trained']}")
                            if 'models' in model_data:
                                logger.info(f"Model has {len(model_data['models'])} trained models")
                                for alg, model in model_data['models'].items():
                                    logger.info(f"  - Model for algorithm {alg} is {'NOT None' if model is not None else 'None'}")
                        else:
                            logger.info(f"Model data is not a dictionary but a {type(model_data)}")
                    except Exception as e:
                        logger.warning(f"Failed to directly inspect model file: {e}")
                else:
                    logger.warning(f"Model file does not exist: {model_path_str}")
            
            # Create the comparison object
            comparison = BaselineComparison(
                baseline_selector,
                meta_optimizer,
                max_evaluations=max_evaluations,
                num_trials=num_trials,
                output_dir=output_dir,
                verbose=True,
                model_path=model_path_str
            )
            logger.info(f"Max evaluations: {max_evaluations}, Num trials: {num_trials}")
            
            # Check if model was loaded successfully
            if hasattr(comparison, 'model_loaded'):
                logger.info(f"Model loaded successfully: {comparison.model_loaded}")
            
            # Run comparison for each benchmark function
            logger.info("Running baseline comparison...")
            for func in benchmark_functions:
                func_name = func.name if hasattr(func, "name") else str(func)
                comparison.run_comparison(
                    problem_name=func_name,
                    problem_func=func,
                    dimensions=dimensions,
                    max_evaluations=max_evaluations,
                    num_trials=num_trials
                )
            
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
                f.write(f"Model Path: {model_path_str}\n")
                f.write(f"Model Loaded Successfully: {comparison.model_loaded}\n")
                f.write(f"Benchmark Functions: {[func.name if hasattr(func, 'name') else str(func) for func in benchmark_functions]}\n")
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
