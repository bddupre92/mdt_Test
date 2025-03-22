"""
MoE Validation Framework Integration Module

This module serves as an integration point between the main_v2.py command-line interface
and the MoE validation framework components implemented in the tests directory.
It also integrates the enhanced validation components:
1. Enhanced Drift Notifications
2. Selective Expert Retraining
3. Continuous Explainability Pipeline
4. Confidence Metrics
"""
import sys
import os
import logging
import argparse
import json
import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path

# Import enhancement components
from core.moe_validation_enhancements import MoEValidationEnhancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_moe_validation(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the MoE validation framework tests
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Command-line arguments
        
    Returns:
    --------
    Dict[str, Any]
        Results of the validation tests
    """
    try:
        # Import MoE validation components
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
        from moe_enhanced_validation_part4 import ExplainableDriftTests
        from moe_validation_runner import main as runner_main
        from moe_interactive_report import generate_interactive_report
        
        # Check for enhanced data configuration - first from environment variable, then from args
        enhanced_data_config = os.environ.get('ENHANCED_DATA_CONFIG') or args.get('enhanced_data_config')
        use_enhanced_data = False
        enhanced_config = {}
        
        if enhanced_data_config:
            logger.info(f"Found enhanced data config path: {enhanced_data_config}")
        
        if enhanced_data_config and os.path.exists(enhanced_data_config):
            logger.info(f"Loading enhanced data configuration from {enhanced_data_config}")
            try:
                # Import enhanced data support module
                from core.enhanced_data_support import load_enhanced_data_config, load_data_pointers
                
                # Load enhanced data configuration
                enhanced_config = load_enhanced_data_config(enhanced_data_config)
                
                if enhanced_config:
                    # Load data pointers
                    data_pointers_file = enhanced_config.get('data_pointers_file')
                    if data_pointers_file and os.path.exists(data_pointers_file):
                        data_pointers = load_data_pointers(data_pointers_file)
                        if data_pointers and data_pointers.get('patients'):
                            logger.info(f"Found {len(data_pointers.get('patients', []))} patients in enhanced data")
                            use_enhanced_data = True
                            
                            # Set visualization directory in args
                            args['visualization_dir'] = enhanced_config.get('visualization_dir', 
                                                                          args.get('results_dir', 'results/moe_validation'))
                            
                            # Add enhanced data to args
                            args['enhanced_data'] = {
                                'config': enhanced_config,
                                'data_pointers': data_pointers
                            }
            except ImportError as e:
                logger.warning(f"Failed to import enhanced data support module: {e}")
            except Exception as e:
                logger.warning(f"Error loading enhanced data configuration: {e}")
        
        # Process command-line arguments
        components = args.get('components', ['all'])
        # Always set interactive to True to ensure reports are generated with timestamps
        interactive = True
        args['interactive'] = True  # Update the args dictionary too
        notify = args.get('notify', False)
        results_dir = args.get('results_dir', '../results/moe_validation')
        explainers = args.get('explainers', ['all'])
        
        # Enhancement feature flags
        enable_retraining = args.get('enable_retraining', False)
        retraining_threshold = args.get('retraining_threshold', 0.3)
        enable_continuous_explain = args.get('enable_continuous_explain', False)
        continuous_explain_interval = args.get('continuous_explain_interval', 60)
        continuous_explain_types = args.get('continuous_explain_types', ['shap', 'feature_importance'])
        enable_confidence = args.get('enable_confidence', False)
        drift_weight = args.get('drift_weight', 0.5)
        confidence_thresholds = args.get('confidence_thresholds', [0.3, 0.5, 0.7, 0.9])
        notify_with_visuals = args.get('notify_with_visuals', False)
        
        # Personalization options
        enable_personalization = args.get('enable_personalization', False)
        adaptation_rate = args.get('adaptation_rate', 0.2)
        profile_update_threshold = args.get('profile_update_threshold', 0.1)
        
        # Ensure results directory exists
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize the enhancement components
        enhancer = MoEValidationEnhancer(results_dir=results_dir)
        
        # Configure enhancer components based on arguments
        if enable_retraining:
            enhancer.retrainer.impact_threshold = retraining_threshold
            logger.info(f"Selective expert retraining enabled with impact threshold {retraining_threshold}")
            
        if enable_continuous_explain:
            enhancer.explainability.explainer_types = continuous_explain_types
            enhancer.explainability.update_interval = continuous_explain_interval
            logger.info(f"Continuous explainability enabled with interval {continuous_explain_interval}s using explainers {continuous_explain_types}")
            
        if enable_confidence:
            enhancer.confidence.drift_weight = drift_weight
            logger.info(f"Confidence metrics enabled with drift weight {drift_weight}")
            
        if notify:
            enhancer.notifier.notify_threshold = args.get('notify_threshold', 0.5)
            enhancer.notifier.with_visuals = notify_with_visuals
            logger.info(f"Enhanced drift notifications enabled with threshold {enhancer.notifier.notify_threshold}")
            
        if enable_personalization:
            enhancer.personalizer.adaptation_rate = adaptation_rate
            enhancer.personalizer.profile_update_threshold = profile_update_threshold
            logger.info(f"Patient profile personalization enabled with adaptation rate {adaptation_rate}")
        
        # Convert components to a list if it's a string
        if isinstance(components, str):
            if components.lower() == 'all':
                components = ['feature_drift', 'explain_drift', 'concept_drift', 'meta_optimizer']
            else:
                components = [component.strip() for component in components.split(',')]
        
        # Create a command-line arguments object compatible with the validation runner
        runner_args = argparse.Namespace()
        runner_args.components = components
        runner_args.interactive = interactive
        runner_args.notify = notify 
        runner_args.results_dir = results_dir
        runner_args.benchmark = args.get('benchmark_comparison', False)
        runner_args.meta_optimizer = args.get('meta_optimizer', False)
        runner_args.meta_learner = args.get('meta_learner', False)
        runner_args.explainers = explainers
        runner_args.notify_threshold = args.get('notify_threshold', 0.5)
        runner_args.enhancer = enhancer
        
        # Run the validation tests
        logger.info(f"Running MoE validation with components: {components}")
        
        # Set system arguments for the runner's parse_args function
        sys_argv = sys.argv
        
        # Construct command-line arguments for the validation runner
        validation_args = ['moe_validation_runner.py']
        
        # Add components
        if components and components != ['all']:
            validation_args.extend(['--components'] + components)
            
        # Add other arguments
        if interactive:
            validation_args.append('--interactive')
        if notify:
            validation_args.append('--notify')
            validation_args.extend(['--notify-threshold', str(args.get('notify_threshold', 0.5))])
            
        # Override sys.argv temporarily
        sys.argv = validation_args
        
        try:
            # Run the validation framework
            result = runner_main()
            
            # If enhancements are enabled, run enhanced validation
            if any([enable_retraining, enable_continuous_explain, enable_confidence, notify]):
                logger.info("Running enhanced validation components...")
                
                # Run enhanced validation with the enhancer
                enhanced_results = enhancer.run_enhanced_validation(args)
                
                # Merge enhanced results with standard results
                if isinstance(result, dict):
                    if 'enhanced_validation' not in result:
                        result['enhanced_validation'] = {}
                    
                    # Add the enhanced results
                    result['enhanced_validation'].update(enhanced_results)
                else:
                    logger.warning("Standard validation results not in expected format, enhanced results stored separately")
                    result = {
                        'standard_validation': result,
                        'enhanced_validation': enhanced_results
                    }
                
                # Save enhanced results to a separate JSON file for easy access
                enhanced_results_path = os.path.join(results_dir, 'enhanced_validation_results.json')
                
                # Convert any non-serializable objects to strings
                def json_serialize(obj):
                    try:
                        return str(obj)
                    except:
                        return "<non-serializable>"
                
                with open(enhanced_results_path, 'w') as f:
                    json.dump(enhanced_results, f, default=json_serialize, indent=2)
                
                logger.info(f"Enhanced validation results saved to {enhanced_results_path}")
        finally:
            # Restore original sys.argv
            sys.argv = sys_argv
            
            # Stop continuous explainability if it was started
            if enable_continuous_explain and hasattr(enhancer, 'explainability'):
                enhancer.stop_continuous_explainability()
        
        # Generate report path from the results
        if isinstance(result, str) and os.path.exists(result):
            report_path = result
        else:
            # Use timestamp for the interactive report filename
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            report_path = os.path.join(results_dir, "reports", f"interactive_report_{timestamp}.html")
            
        # If benchmark comparison was requested, add comparison results
        if args.get('benchmark_comparison', False):
            benchmark_results = add_benchmark_comparison(args)
            if benchmark_results.get('success', False):
                logger.info(f"Benchmark comparison completed: {benchmark_results.get('message')}")
                if 'visualizations' in benchmark_results:
                    logger.info(f"Benchmark visualizations available at: {', '.join(benchmark_results['visualizations'])}")
        
        return {
            "success": True,
            "message": "MoE validation completed successfully",
            "report_path": report_path,
            "exit_code": 0
        }
    
    except Exception as e:
        logger.error(f"Error running MoE validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Error running MoE validation: {str(e)}",
            "exit_code": 1
        }

def add_benchmark_comparison(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add MoE benchmark comparison to evaluate performance against baseline methods
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Command-line arguments
        
    Returns:
    --------
    Dict[str, Any]
        Results of the benchmark comparison
    """
    try:
        # Import required modules
        from baseline_comparison.comparison_runner import BaselineComparison
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Setup benchmark parameters
        dimensions = args.get('dimensions', [2, 5, 10])
        if isinstance(dimensions, str):
            dimensions = [int(d.strip()) for d in dimensions.split(',')]
        
        functions = args.get('functions', ['sphere', 'rosenbrock', 'rastrigin', 'ackley'])
        if isinstance(functions, str):
            functions = [f.strip() for f in functions.split(',')]
        
        optimizers = args.get('optimizers', ['DE', 'ES', 'PSO', 'MoE'])
        if isinstance(optimizers, str):
            optimizers = [o.strip() for o in optimizers.split(',')]
        
        results_dir = args.get('results_dir', '../results/moe_benchmarks')
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize comparison framework
        comparison = BaselineComparison(
            optimizers=optimizers,
            dimensions=dimensions,
            functions=functions,
            results_dir=results_dir
        )
        
        # Run comparison
        logger.info(f"Running benchmark comparison for MoE vs {', '.join([o for o in optimizers if o != 'MoE'])}")
        results = comparison.run_comparison()
        
        # Generate visualizations
        visualizations = []
        for metric in ['best_fitness', 'convergence_rate', 'runtime']:
            viz_path = os.path.join(results_dir, f"moe_benchmark_{metric}.png")
            comparison.visualize_results(metric=metric, output_path=viz_path)
            visualizations.append(viz_path)
        
        # Generate performance profiles
        perf_profile_path = os.path.join(results_dir, "moe_performance_profile.png")
        comparison.generate_performance_profile(output_path=perf_profile_path)
        visualizations.append(perf_profile_path)
        
        return {
            "success": True,
            "message": "MoE benchmark comparison completed successfully",
            "results_dir": results_dir,
            "visualizations": visualizations,
            "exit_code": 0
        }
        
    except Exception as e:
        logger.error(f"Error running MoE benchmark comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Error running MoE benchmark comparison: {str(e)}",
            "exit_code": 1
        }
