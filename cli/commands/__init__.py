"""
Command classes for CLI functionality
"""
from abc import abstractmethod
from typing import Dict, Any, Optional
import argparse
import logging
import os
import sys
from pathlib import Path
import traceback
from baseline_comparison.comparison_runner import BaselineComparison
from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
from baseline_comparison.benchmark_utils import get_benchmark_function, get_all_benchmark_functions
from baseline_comparison.visualization import ComparisonVisualizer
from meta_optimizer.meta import MetaOptimizer
from meta_optimizer.optimizers.optimizer_factory import create_optimizers
from meta_optimizer.benchmark.test_functions import create_test_suite
import matplotlib.pyplot as plt
import numpy as np
import json
from cli.real_data_commands import RealDataValidationCommand
from cli.command_base import Command
from datetime import datetime
from collections import Counter
from cli.problem_wrapper import ProblemWrapper

logger = logging.getLogger(__name__)

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

# Import commands directly when needed to avoid circular imports

def run_baseline_comparison(args):
    """Run baseline comparison command"""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.get('output_dir') or f"results/baseline_validation/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize baseline selector and meta optimizer
    dimensions = args.get('dimensions', 30)
    max_evaluations = args.get('max_evaluations', 10000)
    num_trials = args.get('num_trials', 30)

    # Load optimizer implementations first
    try:
        optimizers = create_optimizers()
        logger.info(f"Loaded {len(optimizers)} optimizers: {list(optimizers.keys())}")
    except ImportError as e:
        logger.error(f"Could not import optimizer implementations: {e}")
        raise

    # 1. Simple Baseline (Random Selection)
    from baseline_comparison.baseline_algorithms.simple_baseline import SimpleBaselineSelector
    simple_baseline = SimpleBaselineSelector()
    simple_baseline.set_available_algorithms(list(optimizers.keys()))
    logger.info("Initialized SimpleBaselineSelector")
    
    # 2. Meta-Learner (Basic Bandit)
    meta_learner = MetaOptimizer(
        dim=dimensions,
        bounds=[(-5, 5)] * dimensions,  # Default bounds, will be overridden per problem
        optimizers=optimizers,
        n_parallel=1,
        visualize_selection=True,
        default_max_evals=max_evaluations
    )
    logger.info("Initialized Meta-Learner")
    
    # 3. Enhanced Meta-Optimizer
    enhanced_meta = MetaOptimizer(
        dim=dimensions,
        bounds=[(-5, 5)] * dimensions,  # Default bounds, will be overridden per problem
        optimizers=optimizers,
        n_parallel=1,
        visualize_selection=True,
        use_ml_selection=True,
        default_max_evals=max_evaluations
    )
    logger.info("Initialized Enhanced Meta-Optimizer")
    
    # 4. SATzilla-Inspired Selector
    satzilla_selector = SatzillaInspiredSelector()
    satzilla_selector.set_available_algorithms(list(optimizers.keys()))
    logger.info("Initialized SatzillaInspiredSelector")
    
    # Load SATzilla model if provided
    model_path = args.get('selector_path')
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
            raise

    # Run comparison
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
    
    # Get functions to test
    if args.get('all_functions', False):
        logger.info(f"Using all benchmark functions")
        functions = ["sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "levy", "schwefel"]
    else:
        functions = args.get('functions', ['sphere', 'rosenbrock'])
        if isinstance(functions, str):
            functions = functions.split(",")
        logger.info(f"Using specified functions: {functions}")
    
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
            
            # Also save using the selector's save_model method for compatibility with BaselineComparisonCommand
            selector_path = model_dir / "satzilla_selector.joblib"
            selector.save_model(str(selector_path))
            self.logger.info(f"Trained selector also saved to {selector_path} (compatible format)")
            
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

class BaselineComparisonCommand(Command):
    """Command for running baseline comparison"""
    
    def execute(self) -> int:
        """
        Execute the baseline comparison command
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            run_baseline_comparison(self.args)
            return 0
        except Exception as e:
            self.logger.error(f"Error during baseline comparison: {e}")
            traceback.print_exc()
            return 1

class RealDataValidationCommand(Command):
    """Command for integrating and validating real clinical data with the MoE framework"""
    
    def execute(self) -> int:
        """
        Execute real data validation and integration with MoE
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Import necessary components
            from data_integration.clinical_data_adapter import ClinicalDataAdapter
            from data_integration.clinical_data_validator import ClinicalDataValidator
            from data_integration.real_synthetic_comparator import RealSyntheticComparator
            from utils.enhanced_synthetic_data import EnhancedSyntheticDataGenerator
            import pandas as pd
            import numpy as np
            from datetime import datetime
            import json

            self.logger.info("Starting real data validation...")
            
            # Extract arguments
            clinical_data_path = self.args.clinical_data
            data_format = self.args.data_format
            config_path = self.args.config
            target_column = self.args.target_column
            output_dir = self.args.output_dir
            synthetic_compare = self.args.synthetic_compare
            drift_type = self.args.drift_type
            run_mode = self.args.run_mode
            
            # Create output directory structure
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
            
            # Generate timestamp for this run
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            
            # Initialize adapter and validator
            adapter = ClinicalDataAdapter(config_path)
            validator = ClinicalDataValidator(config_path)
            
            # Load data
            self.logger.info(f"Loading clinical data from {clinical_data_path}...")
            try:
                clinical_data = adapter.load_data(clinical_data_path, data_format)
                self.logger.info(f"Loaded {len(clinical_data)} records with {len(clinical_data.columns)} features.")
            except Exception as e:
                self.logger.error(f"Error loading clinical data: {str(e)}")
                return 1
            
            # Validate data
            self.logger.info("Validating clinical data quality and compatibility...")
            validation_report = validator.validate_all(
                clinical_data, 
                save_path=os.path.join(output_dir, 'reports', f'validation_report_{timestamp}.json')
            )
            
            # Preprocess data
            self.logger.info("Preprocessing clinical data...")
            processed_data = adapter.preprocess(clinical_data)
            
            # Save processed data
            processed_path = os.path.join(output_dir, 'data', f'processed_clinical_{timestamp}.csv')
            processed_data.to_csv(processed_path, index=False)
            
            # Generate synthetic data for comparison if requested
            synthetic_data = None
            if synthetic_compare or run_mode in ['full', 'comparison']:
                self.logger.info("Generating comparable synthetic data...")
                
                # Generate synthetic data with similar feature distributions
                generator = EnhancedSyntheticDataGenerator(
                    num_samples=len(clinical_data),
                    drift_type=drift_type,
                    data_modality='mixed'  # Generate mixed data type
                )
                
                synthetic_data = generator.generate_mirrored_data(
                    feature_stats={},  # This would be extracted from clinical_data
                    target_column=target_column,
                    target_ratio=0.3  # Default value
                )
                
                # Save synthetic data
                synthetic_path = os.path.join(output_dir, 'data', f'synthetic_comparison_{timestamp}.csv')
                synthetic_data.to_csv(synthetic_path, index=False)
            
            # Run MoE validation if requested
            if run_mode in ['full', 'validation']:
                self.logger.info("Running MoE validation with real clinical data...")
                
                # Prepare output directory for validation
                validation_dir = os.path.join(output_dir, 'moe_validation')
                os.makedirs(validation_dir, exist_ok=True)
                
                # Import MoE validation components
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tests'))
                
                try:
                    from moe_enhanced_validation_part4 import run_enhanced_validation
                    from moe_interactive_report import generate_interactive_report
                    
                    # Create expert types mapping based on column names
                    expert_mapping = {}
                    for col in clinical_data.columns:
                        col_lower = col.lower()
                        if any(term in col_lower for term in ['heart', 'rate', 'temp', 'blood']):
                            expert_mapping[col] = 'physiological'
                        elif any(term in col_lower for term in ['weather', 'humid', 'pressure']):
                            expert_mapping[col] = 'environmental'
                        elif any(term in col_lower for term in ['stress', 'sleep', 'activity']):
                            expert_mapping[col] = 'behavioral'
                        elif any(term in col_lower for term in ['medication', 'drug', 'treatment']):
                            expert_mapping[col] = 'medication'
                        else:
                            expert_mapping[col] = 'general'
                    
                    # Save expert mapping for reference
                    mapping_path = os.path.join(validation_dir, f'expert_mapping_{timestamp}.json')
                    with open(mapping_path, 'w') as f:
                        json.dump(expert_mapping, f, indent=2)
                    
                    # Run enhanced validation
                    validation_results = run_enhanced_validation(
                        real_data=clinical_data,
                        synthetic_data=synthetic_data,
                        drift_type=drift_type,
                        output_dir=validation_dir,
                        expert_mapping=expert_mapping,
                        target_column=target_column
                    )
                    
                    # Generate interactive report
                    if hasattr(self.args, 'interactive_report') and self.args.interactive_report:
                        report_path = os.path.join(validation_dir, f'real_data_report_{timestamp}.html')
                        generate_interactive_report(
                            results=validation_results,
                            output_path=report_path,
                            title="Real Clinical Data MoE Validation Report",
                            description=f"Validation results for real clinical data from {clinical_data_path}",
                            metadata={
                                "timestamp": timestamp,
                                "data_source": clinical_data_path,
                                "num_records": len(clinical_data),
                                "num_features": len(clinical_data.columns),
                                "target_column": target_column,
                                "drift_type": drift_type
                            }
                        )
                        
                        self.logger.info(f"MoE validation complete. Interactive report saved to: {report_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error running MoE validation: {str(e)}")
                    traceback.print_exc()
                    return 1
            
            self.logger.info(f"Real data validation completed successfully.")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error during real data validation: {str(e)}")
            traceback.print_exc()
            return 1

class MoEValidationCommand(Command):
    """Command for running the MoE validation framework"""
    
    def execute(self) -> int:
        """
        Execute the MoE validation framework tests
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Import the MoE validation module
            from core.moe_validation import run_moe_validation
            
            # Create a dictionary with the args from self.args
            args_dict = {}
            # Components
            args_dict['components'] = getattr(self.args, 'components', ['all'])
            # Interactive
            args_dict['interactive'] = getattr(self.args, 'interactive', False)
            
            # Basic arguments
            args_dict['results_dir'] = getattr(self.args, 'results_dir', 'results/moe_validation')
            args_dict['benchmark_comparison'] = getattr(self.args, 'benchmark_comparison', False)
            args_dict['explainers'] = getattr(self.args, 'explainers', ['all'])
            
            # Enhanced Drift Notifications arguments
            args_dict['notify'] = getattr(self.args, 'notify', False)
            args_dict['notify_threshold'] = getattr(self.args, 'notify_threshold', 0.5)
            args_dict['notify_with_visuals'] = getattr(self.args, 'notify_with_visuals', False)
            
            # Selective Expert Retraining arguments
            args_dict['enable_retraining'] = getattr(self.args, 'enable_retraining', False)
            args_dict['retraining_threshold'] = getattr(self.args, 'retraining_threshold', 0.3)
            
            # Continuous Explainability arguments
            args_dict['enable_continuous_explain'] = getattr(self.args, 'enable_continuous_explain', False)
            args_dict['continuous_explain_interval'] = getattr(self.args, 'continuous_explain_interval', 60)
            args_dict['continuous_explain_types'] = getattr(self.args, 'continuous_explain_types', ['shap', 'feature_importance'])
            
            # Confidence Metrics arguments
            args_dict['enable_confidence'] = getattr(self.args, 'enable_confidence', False)
            args_dict['drift_weight'] = getattr(self.args, 'drift_weight', 0.5)
            args_dict['confidence_thresholds'] = getattr(self.args, 'confidence_thresholds', [0.3, 0.5, 0.7, 0.9])
            
            # Execute the MoE validation framework
            self.logger.info("Running MoE validation framework tests...")
            self.logger.info(f"Components: {args_dict['components']}")
            result = run_moe_validation(args_dict)
            
            # Check if validation was successful
            if result.get('success', False):
                self.logger.info(result.get('message', 'MoE validation completed successfully'))
                
                # Log report path if available
                if 'report_path' in result and result['report_path']:
                    self.logger.info(f"Validation report available at: {result['report_path']}")
                    
                return result.get('exit_code', 0)
            else:
                self.logger.error(result.get('message', 'MoE validation failed'))
                return result.get('exit_code', 1)
                
        except Exception as e:
            self.logger.error(f"Error during MoE validation: {e}")
            traceback.print_exc()
            return 1

# Define COMMAND_MAP here, AFTER the BaselineComparisonCommand class is defined
# Command map only includes commands defined directly in this module or already imported
COMMAND_MAP = {
    "baseline_comparison": BaselineComparisonCommand,
    "train_satzilla": SatzillaTrainingCommand,
    "dynamic_optimization": DynamicOptimizationCommand,
    "moe_validation": MoEValidationCommand,
    "real_data_validation": RealDataValidationCommand,
    # The real_data_validation command is implemented in its own module to avoid circular imports
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
    
    # Dynamic optimization
    if hasattr(args, 'dynamic_optimization') and args.dynamic_optimization:
        return DynamicOptimizationCommand(args)
        
    return None