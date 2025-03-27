import argparse
import sys
from typing import List, Optional, Dict, Any

def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser with all argument groups
    
    Returns:
    --------
    argparse.ArgumentParser
        The configured argument parser
    """
    parser = argparse.ArgumentParser(description='Advanced Optimization Framework')
    
    # Add all argument groups
    add_general_args(parser)
    add_meta_learning_args(parser)
    add_optimization_args(parser)
    add_evaluation_args(parser)
    add_explainability_args(parser)
    add_drift_detection_args(parser)
    add_migraine_args(parser)
    add_benchmark_args(parser)
    add_moe_validation_args(parser)
    # Note: add_moe_comparison_args is not called here because those arguments
    # are added to the moe_comparison_parser subparser directly
    add_real_data_validation_args(parser)
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to execute",
        required=True
    )
    
    # Define a baseline comparison command
    baseline_parser = subparsers.add_parser(
        "baseline_comparison",
        help="Run baseline comparison benchmark"
    )
    
    # Define the MoE validation command
    moe_validation_parser = subparsers.add_parser(
        "moe_validation",
        help="Run MoE validation framework tests"
    )
    
    # Define the MoE comparison command
    moe_comparison_parser = subparsers.add_parser(
        "moe_comparison",
        help="Run comparison between MoE and baseline approaches"
    )
    
    # Add MoE comparison specific arguments 
    add_moe_comparison_args(moe_comparison_parser)
    
    return parser

def add_general_args(parser: argparse.ArgumentParser) -> None:
    """
    Add general arguments to the parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    parser.add_argument('--config', help="Path to configuration file")
    parser.add_argument('--verbose', '-v', action='count', default=0, 
                      help="Increase verbosity (can be used multiple times)")
    parser.add_argument('--quiet', '-q', action='store_true',
                      help="Suppress non-error messages")
    parser.add_argument('--log-file', help="Path to log file")
    parser.add_argument('--seed', type=int, help="Random seed for reproducibility")
    parser.add_argument('--visualize', action='store_true', help="Enable visualization")
    parser.add_argument('--summary', action='store_true', help="Show summary")
    parser.add_argument('--export-dir', help="Directory for exporting data", default="results")

def add_meta_learning_args(parser: argparse.ArgumentParser) -> None:
    """
    Add meta-learning related arguments to the parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    # Meta-learning flags
    parser.add_argument('--meta', help="Run meta-learning", action='store_true')
    parser.add_argument('--enhanced-meta', help="Run enhanced meta-optimizer with feature extraction and ML-based selection", action='store_true')
    
    # Meta-learning parameters
    parser.add_argument('--meta-method', choices=['random', 'ucb', 'thompson', 'bayesian'], default='ucb', 
                        help="Meta-learning method")
    parser.add_argument('--meta-surrogate', choices=['gp', 'rf', 'mlp'], default='gp', 
                        help="Surrogate model for meta-learning")
    parser.add_argument('--meta-selection', choices=['greedy', 'probability'], default='greedy', 
                        help="Selection strategy for meta-learning")
    parser.add_argument('--meta-exploration', type=float, default=0.2, 
                        help="Exploration parameter for meta-learning")
    parser.add_argument('--meta-history-weight', type=float, default=0.5, 
                        help="History weight for meta-learning")
    
    # Enhanced meta learning parameters
    parser.add_argument('--use-ml-selection', action='store_true', 
                        help="Use machine learning for algorithm selection")
    parser.add_argument('--extract-features', action='store_true', 
                        help="Extract problem features")
    parser.add_argument('--dimension', type=int, default=10, 
                        help="Dimension for meta-learning problems")

def add_optimization_args(parser: argparse.ArgumentParser) -> None:
    """
    Add optimization related arguments to the parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    parser.add_argument('--optimize', help="Run optimization", action='store_true')
    parser.add_argument('--compare-optimizers', help="Compare optimizers", action='store_true')
    parser.add_argument('--optimizer', choices=['PSO', 'DE', 'ACO', 'GWO', 'ES', 'SA', 'GA'], default='DE',
                        help="Optimizer to use")
    parser.add_argument('--max-evaluations', type=int, default=1000,
                        help="Maximum function evaluations")
    parser.add_argument('--population-size', type=int, default=50,
                        help="Population size for population-based optimizers")
    parser.add_argument('--n-iterations', type=int, default=100,
                        help="Number of iterations")
    parser.add_argument('--problem', choices=['sphere', 'rosenbrock', 'rastrigin', 'ackley'], default='sphere',
                        help="Test problem to optimize")
    parser.add_argument('--n-parallel', type=int, default=1,
                        help="Number of parallel optimizers to run")
    parser.add_argument('--budget-per-iteration', type=int, default=50,
                        help="Budget per iteration")

def add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    """
    Add evaluation related arguments to the parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    parser.add_argument('--evaluate', help="Evaluate model", action='store_true')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help="Test set size for evaluation")
    parser.add_argument('--n-folds', type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument('--random-seed', type=int, default=42,
                        help="Random seed for data splitting")
    parser.add_argument('--metrics', choices=['accuracy', 'f1', 'auc', 'precision', 'recall', 'all'], 
                        default='all', help="Evaluation metrics")

def add_explainability_args(parser: argparse.ArgumentParser) -> None:
    """
    Add explainability related arguments to the parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    parser.add_argument('--explain', help="Run explainability analysis", action='store_true')
    parser.add_argument('--auto-explain', help="Run automatic explainability analysis", action='store_true')
    parser.add_argument('--explainer', choices=['shap', 'lime', 'eli5'], default='shap',
                       help="Explainer method")
    parser.add_argument('--explain-plots', help="Generate explainability plots", action='store_true')
    parser.add_argument('--explain-plot-types', 
                       choices=['summary', 'dependence', 'force', 'decision', 'all'],
                       default='all', help="Types of explainability plots")
    parser.add_argument('--explain-samples', type=int, default=5,
                       help="Number of samples to explain")

def add_drift_detection_args(parser: argparse.ArgumentParser) -> None:
    """
    Add drift detection related arguments to the parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    parser.add_argument('--drift', help="Run drift detection", action='store_true')
    parser.add_argument('--run-meta-learner-with-drift', help="Run meta-learner with drift detection", action='store_true')
    parser.add_argument('--explain-drift', help="Explain drift detection", action='store_true')
    parser.add_argument('--drift-window', type=int, default=50,
                      help="Window size for drift detection")
    parser.add_argument('--drift-threshold', type=float, default=0.5,
                      help="Threshold for drift detection")
    parser.add_argument('--significance-level', type=float, default=0.05,
                      help="Significance level for drift detection")
    parser.add_argument('--min-drift-interval', type=int, default=30,
                      help="Minimum interval between drift points")
    parser.add_argument('--ema-alpha', type=float, default=0.3,
                      help="Alpha parameter for EMA in drift detection")

def add_migraine_args(parser: argparse.ArgumentParser) -> None:
    """
    Add migraine specific arguments to the parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    parser.add_argument('--import-migraine-data', help="Import migraine data", action='store_true')
    parser.add_argument('--predict-migraine', help="Predict migraine", action='store_true')
    parser.add_argument('--migraine-period-days', type=int, default=7,
                     help="Period in days for migraine prediction")
    parser.add_argument('--migraine-features', 
                     choices=['basic', 'enhanced', 'all'], default='all',
                     help="Feature set for migraine prediction")
    parser.add_argument('--migraine-model', 
                     choices=['lr', 'rf', 'xgb', 'lstm', 'transformer'], default='rf',
                     help="Model for migraine prediction")
    parser.add_argument('--migraine-data-path', type=str,
                     help="Path to migraine data")
    parser.add_argument('--migraine-output-path', type=str,
                     help="Path for migraine prediction output")

def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """
    Add benchmark arguments to the parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    parser.add_argument('--run-benchmark', help="Run benchmark", action='store_true')
    parser.add_argument('--benchmark-problems', 
                     choices=['all', 'classic', 'cec2013', 'cec2017'], default='classic',
                     help="Benchmark problem set")
    parser.add_argument('--benchmark-repetitions', type=int, default=30,
                     help="Number of repetitions for benchmarking")
    parser.add_argument('--benchmark-record-trajectory', action='store_true',
                     help="Record optimization trajectory during benchmarking")
    parser.add_argument('--benchmark-optimizers', 
                     help="Comma-separated list of optimizers to benchmark")
    parser.add_argument('--model-path', type=str,
                     help="Path to trained SatzillaInspiredSelector model")
    
    # New baseline comparison arguments
    parser.add_argument('--baseline-comparison', help="Run baseline comparison", action='store_true')
    parser.add_argument('--dimensions', type=int, default=3,
                     help="Dimension for optimization problems")
    # Note: --max-evaluations is defined in add_optimization_args to avoid duplication
    parser.add_argument('--num-trials', type=int, default=3,
                     help="Number of trials to run per algorithm")
    parser.add_argument('--functions', type=str, default='sphere',
                     help="Comma-separated list of test functions (sphere,rosenbrock,rastrigin,ackley,griewank,levy,schwefel)")
    parser.add_argument('--benchmark-output-dir', type=str, default='results',
                     help="Directory to save benchmark results")
    parser.add_argument('--selector-path', type=str,
                     help="Path to trained SatzillaInspiredSelector model")
    parser.add_argument('--all-functions', action='store_true',
                     help="Use all available benchmark functions")
    
    # Dynamic optimization visualization
    parser.add_argument('--dynamic-optimization', action='store_true',
                     help="Run dynamic optimization visualization")
    parser.add_argument('--function', type=str, 
                     choices=['sphere', 'rastrigin', 'rosenbrock', 'ackley', 'griewank', 'levy', 'schwefel'],
                     help="Test function to use for dynamic optimization")
    parser.add_argument('--dynamic-drift-type', type=str, 
                     choices=['sudden', 'oscillatory', 'linear', 'incremental', 'gradual', 'random', 'noise'],
                     help="Type of drift for dynamic optimization")
    parser.add_argument('--drift-rate', type=float, default=0.1,
                     help="Rate of drift (0.0 to 1.0)")
    parser.add_argument('--drift-interval', type=int, default=20,
                     help="Interval between drift events (in function evaluations)")
    parser.add_argument('--severity', type=float, default=1.0,
                     help="Severity of drift (0.0 to 1.0)")
    parser.add_argument('--max-iterations', type=int, default=500,
                     help="Maximum number of iterations for dynamic optimization")
    parser.add_argument('--reoptimize-interval', type=int, default=50,
                     help="Re-optimize after this many function evaluations")
    parser.add_argument('--show-plot', action='store_true',
                     help="Show plot in addition to saving it")

def add_moe_validation_args(parser: argparse.ArgumentParser) -> None:
    """
    Add MoE validation framework related arguments to the parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    parser.add_argument('--moe-validation', action='store_true',
                      help="Run MoE validation framework tests")
    parser.add_argument('--components', nargs='+', 
                      choices=['meta_optimizer', 'meta_learner', 'drift', 'explainability', 'gating', 'integrated', 'explain_drift', 'all'],
                      default=['all'], 
                      help="Components to test in MoE validation framework")
    parser.add_argument('--interactive', action='store_true',
                      help="Generate interactive HTML report with visualizations")
    
    # Basic validation arguments
    parser.add_argument('--results-dir', default='results/moe_validation',
                      help="Directory to save validation results")
    parser.add_argument('--benchmark-comparison', action='store_true',
                      help="Run benchmarking against baseline methods for MoE performance")
    parser.add_argument('--explainers', nargs='+',
                      choices=['shap', 'lime', 'feature_importance', 'optimizer'],
                      default=['shap'],
                      help="Explainability methods to use in validation")
    
    # Enhanced Drift Notifications
    notification_group = parser.add_argument_group('Enhanced Drift Notifications')
    notification_group.add_argument('--notify', action='store_true',
                     help="Enable enhanced notifications with explanations of why drift occurred")
    notification_group.add_argument('--notify-threshold', type=float, default=0.5,
                     help="Threshold for drift notifications (0-1, higher means only notify on severe drift)")
    notification_group.add_argument('--notify-with-visuals', action='store_true',
                     help="Include visualizations with drift notifications")
    
    # Selective Expert Retraining
    retraining_group = parser.add_argument_group('Selective Expert Retraining')
    retraining_group.add_argument('--enable-retraining', action='store_true',
                    help="Enable selective retraining of experts affected by drift")
    retraining_group.add_argument('--retraining-threshold', type=float, default=0.3,
                    help="Impact threshold for expert retraining (0-1)")
    
    # Continuous Explainability
    continuous_group = parser.add_argument_group('Continuous Explainability')
    continuous_group.add_argument('--enable-continuous-explain', action='store_true',
                     help="Enable continuous explainability monitoring")
    continuous_group.add_argument('--continuous-explain-interval', type=int, default=60,
                     help="Update interval in seconds for continuous explainability")
    continuous_group.add_argument('--continuous-explain-types', nargs='+',
                     choices=['shap', 'lime', 'feature_importance', 'optimizer'],
                     default=['shap', 'feature_importance'],
                     help="Explainer types for continuous monitoring")
    
    # Confidence Metrics
    confidence_group = parser.add_argument_group('Confidence Metrics')
    confidence_group.add_argument('--enable-confidence', action='store_true',
                     help="Enable confidence metrics integrating drift severity with model uncertainty")
    confidence_group.add_argument('--drift-weight', type=float, default=0.5,
                     help="Weight of drift impact in confidence calculation (0-1)")
    confidence_group.add_argument('--confidence-thresholds', nargs='+', type=float,
                     default=[0.3, 0.5, 0.7, 0.9],
                     help="Confidence thresholds for reporting (space-separated values between 0-1)")
    
    # Enhanced Synthetic Data
    enhanced_data_group = parser.add_argument_group('Enhanced Synthetic Data')
    enhanced_data_group.add_argument('--enhanced-data-config', type=str,
                        help="Path to enhanced synthetic data configuration file")
    enhanced_data_group.add_argument('--include-drift-visualizations', action='store_true',
                        help="Include drift visualizations from enhanced synthetic data in reports")
    enhanced_data_group.add_argument('--use-enhanced-data-for-validation', action='store_true',
                        help="Use enhanced synthetic data for MoE validation test cases")

def add_moe_comparison_args(parser: argparse.ArgumentParser) -> None:
    """
    Add MoE comparison related arguments to the parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    parser.add_argument('--moe-comparison', action='store_true',
                      help="Run comparison between MoE and baseline approaches")
    parser.add_argument('--moe-config-path', type=str, default=None,
                      help="Path to MoE configuration file (JSON format)")
    parser.add_argument('--moe-model-path', type=str, default=None,
                      help="Path to pre-trained MoE model (optional)")
    parser.add_argument('--output-dir', type=str, default="results/moe_comparison",
                      help="Directory to save MoE comparison results")
    parser.add_argument('--num-trials', '-t', type=int, default=10, 
                      help="Number of trials per function (default: 10)")
    parser.add_argument('--functions', type=str, nargs="+",
                      default=["sphere", "rosenbrock", "rastrigin", "ackley", "griewank"],
                      help="List of benchmark functions to use in comparison")
    parser.add_argument('--all-functions', action="store_true",
                      help="Use all available benchmark functions for comparison")
    parser.add_argument('--visualize-moe-contributions', action='store_true',
                      help="Generate visualizations showing expert contributions in MoE")
    parser.add_argument('--calculate-expert-impact', action='store_true',
                      help="Calculate and report the impact of each expert on final predictions")
    parser.add_argument('--detailed-report', action='store_true',
                      help="Generate a detailed HTML report with interactive visualizations")
    parser.add_argument('--include-confidence-metrics', action='store_true',
                      help="Include confidence metrics in the comparison results")

def add_real_data_validation_args(parser: argparse.ArgumentParser) -> None:
    """
    Add real clinical data validation related arguments to the parser
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The parser to add arguments to
    """
    # Main group for real data validation
    parser.add_argument('--real-data-validation', action='store_true',
                      help="Run real clinical data validation with MoE framework")
    
    # Data source arguments
    data_group = parser.add_argument_group('Real Data Sources')
    data_group.add_argument('--clinical-data', type=str,
                      help="Path to clinical data file (CSV, JSON, or Excel)")
    data_group.add_argument('--data-format', type=str, default='csv',
                      choices=['csv', 'json', 'excel', 'parquet'],
                      help="Format of clinical data file")
    data_group.add_argument('--data-config', type=str,
                      help="Path to configuration file for data integration")
    data_group.add_argument('--target-column', type=str, default='migraine',
                      help="Target column name in clinical data")
    
    # Validation options
    validation_group = parser.add_argument_group('Validation Options')
    validation_group.add_argument('--validation-output-dir', type=str, default='results/real_data_validation',
                      help="Directory to store validation results")
    validation_group.add_argument('--synthetic-compare', action='store_true',
                      help="Generate and compare with synthetic data")
    validation_group.add_argument('--validation-drift-type', type=str, default='sudden',
                      choices=['sudden', 'gradual', 'recurring', 'none'],
                      help="Type of drift to analyze")
    validation_group.add_argument('--run-mode', type=str, default='full',
                      choices=['full', 'validation', 'comparison', 'report'],
                      help="Which parts of the process to run")
    
    # Integration with MoE framework
    integration_group = parser.add_argument_group('MoE Integration')
    integration_group.add_argument('--interactive-report', action='store_true',
                      help="Generate interactive HTML report with MoE validation results")
    integration_group.add_argument('--expert-mapping-method', type=str, default='auto',
                      choices=['auto', 'manual', 'config'],
                      help="Method to map features to expert types")

def parse_args(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parse command-line arguments
    
    Args:
        args: Command-line arguments (if None, use sys.argv)
        
    Returns:
        Dictionary of parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Advanced Optimization Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add global arguments that apply to all commands
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error messages"
    )
    
    parser.add_argument(
        "--log-file",
        help="Path to log file"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to execute",
        required=True
    )
    
    # Define a baseline comparison command
    baseline_parser = subparsers.add_parser(
        "baseline_comparison",
        help="Run baseline comparison benchmark"
    )
    
    # Define the MoE validation command
    moe_validation_parser = subparsers.add_parser(
        "moe_validation",
        help="Run MoE validation framework tests"
    )
    
    # Define the MoE comparison command
    moe_comparison_parser = subparsers.add_parser(
        "moe_comparison",
        help="Run comparison between MoE and baseline approaches"
    )
    
    # Add MoE comparison specific arguments
    moe_comparison_parser.add_argument(
        "--moe-config-path",
        type=str,
        help="Path to MoE configuration file (JSON format)"
    )
    
    moe_comparison_parser.add_argument(
        "--moe-model-path",
        type=str,
        help="Path to pre-trained MoE model (optional)"
    )
    
    moe_comparison_parser.add_argument(
        "--output-dir",
        type=str,
        default="results/moe_comparison",
        help="Directory to save MoE comparison results"
    )
    
    moe_comparison_parser.add_argument(
        "--num-trials", "-t",
        type=int,
        default=10,
        help="Number of trials per function (default: 10)"
    )
    
    moe_comparison_parser.add_argument(
        "--functions",
        type=str,
        nargs="+",
        default=["sphere", "rosenbrock", "rastrigin", "ackley", "griewank"],
        help="List of benchmark functions to use in comparison"
    )
    
    moe_comparison_parser.add_argument(
        "--all-functions", 
        action="store_true",
        help="Use all available benchmark functions for comparison"
    )
    
    moe_comparison_parser.add_argument(
        "--visualize-moe-contributions",
        action="store_true",
        help="Generate visualizations showing expert contributions in MoE"
    )
    
    moe_comparison_parser.add_argument(
        "--calculate-expert-impact",
        action="store_true",
        help="Calculate and report the impact of each expert on final predictions"
    )
    
    moe_comparison_parser.add_argument(
        "--detailed-report",
        action="store_true",
        help="Generate a detailed HTML report with interactive visualizations"
    )
    
    moe_comparison_parser.add_argument(
        "--include-confidence-metrics",
        action="store_true",
        help="Include confidence metrics in the comparison results"
    )
    
    # Define the real data validation command
    real_data_parser = subparsers.add_parser(
        "real_data_validation",
        help="Run real clinical data validation with MoE framework"
    )
    
    # Add real data validation arguments
    real_data_parser.add_argument(
        "--clinical-data", "-c",
        type=str,
        required=True,
        help="Path to clinical data file (CSV, JSON, or Excel)"
    )
    
    real_data_parser.add_argument(
        "--data-format",
        type=str,
        default="csv",
        choices=["csv", "json", "excel", "parquet"],
        help="Format of clinical data file"
    )
    
    real_data_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file for data integration"
    )
    
    real_data_parser.add_argument(
        "--target-column",
        type=str,
        default="migraine",
        help="Target column name in clinical data"
    )
    
    real_data_parser.add_argument(
        "--output-dir",
        type=str,
        default="results/real_data_validation",
        help="Directory to store validation results"
    )
    
    real_data_parser.add_argument(
        "--synthetic-compare",
        action="store_true",
        help="Generate and compare with synthetic data"
    )
    
    real_data_parser.add_argument(
        "--drift-type",
        type=str,
        default="sudden",
        choices=["sudden", "gradual", "recurring", "none"],
        help="Type of drift to analyze"
    )
    
    real_data_parser.add_argument(
        "--run-mode",
        type=str,
        default="full",
        choices=["full", "validation", "comparison", "report"],
        help="Which parts of the process to run"
    )
    
    real_data_parser.add_argument(
        "--interactive-report",
        action="store_true",
        help="Generate interactive HTML report with MoE validation results"
    )

    # Add baseline comparison arguments
    baseline_parser.add_argument(
        "--dimensions", "-d",
        type=int,
        default=2,
        help="Number of dimensions for benchmark functions"
    )
    
    baseline_parser.add_argument(
        "--max-evaluations", "-e",
        type=int,
        default=1000,
        help="Maximum number of function evaluations per algorithm"
    )
    
    baseline_parser.add_argument(
        "--num-trials", "-t",
        type=int,
        default=3,
        help="Number of trials to run for statistical significance"
    )
    
    baseline_parser.add_argument(
        "--functions", "-f",
        nargs="+",
        default=["sphere", "rosenbrock"],
        help="Benchmark functions to use"
    )
    
    baseline_parser.add_argument(
        "--all-functions",
        action="store_true",
        help="Use all available benchmark functions"
    )
    
    baseline_parser.add_argument(
        "--output-dir", "-o",
        default="results/baseline_comparison",
        help="Output directory for results"
    )
    
    baseline_parser.add_argument(
        "--selector-path", "-s",
        help="Path to a trained SATzilla-inspired algorithm selector"
    )
    
    baseline_parser.add_argument(
        "--timestamp-dir",
        action="store_true",
        default=True,
        help="Create a timestamped subdirectory for results"
    )
    
    baseline_parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable visualization generation"
    )
    
    baseline_parser.add_argument(
        "--equal-budget",
        action="store_true",
        help="Use equal budget for both baseline and meta-optimizer for fair comparison"
    )
    
    # Define a SATzilla training command
    satzilla_train_parser = subparsers.add_parser(
        "train_satzilla",
        help="Train a SATzilla-inspired algorithm selector"
    )
    
    # Add MoE validation framework arguments
    moe_validation_parser.add_argument(
        "--components", 
        nargs="+",
        choices=["meta_optimizer", "meta_learner", "drift", "explainability", "gating", "integrated", "explain_drift", "all"],
        default=["all"],
        help="Components to test in MoE validation framework"
    )
    
    moe_validation_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive HTML report with visualizations"
    )
    
    moe_validation_parser.add_argument(
        "--notify",
        action="store_true",
        help="Enable automatic drift notifications"
    )
    
    moe_validation_parser.add_argument(
        "--notify-threshold",
        type=float,
        default=0.5,
        help="Threshold for drift notifications (0-1, higher means only notify on severe drift)"
    )
    
    moe_validation_parser.add_argument(
        "--results-dir",
        default="results/moe_validation",
        help="Directory to save validation results"
    )
    
    moe_validation_parser.add_argument(
        "--benchmark-comparison",
        action="store_true",
        help="Run benchmarking against baseline methods for MoE performance"
    )
    
    moe_validation_parser.add_argument(
        "--explainers",
        nargs="+",
        choices=["shap", "lime", "feature_importance", "optimizer_explainer", "all"],
        default=["all"],
        help="Explainers to use for MoE explainability tests"
    )
    
    # Enhanced Drift Notifications arguments
    notification_group = moe_validation_parser.add_argument_group('Enhanced Drift Notifications')
    notification_group.add_argument(
        "--notify-with-visuals",
        action="store_true",
        help="Include visualizations with drift notifications"
    )
    
    # Selective Expert Retraining arguments
    retraining_group = moe_validation_parser.add_argument_group('Selective Expert Retraining')
    retraining_group.add_argument(
        "--enable-retraining",
        action="store_true",
        help="Enable selective retraining of experts affected by drift"
    )
    retraining_group.add_argument(
        "--retraining-threshold",
        type=float,
        default=0.3,
        help="Impact threshold to determine which experts need retraining (0-1)"
    )
    
    # Continuous Explainability arguments
    continuous_group = moe_validation_parser.add_argument_group('Continuous Explainability')
    continuous_group.add_argument(
        "--enable-continuous-explain",
        action="store_true",
        help="Enable continuous explainability monitoring"
    )
    continuous_group.add_argument(
        "--continuous-explain-interval",
        type=int,
        default=60,
        help="Update interval in seconds for continuous explainability"
    )
    continuous_group.add_argument(
        "--continuous-explain-types",
        nargs="+",
        choices=["shap", "lime", "feature_importance", "optimizer"],
        default=["shap", "feature_importance"],
        help="Explainer types for continuous monitoring"
    )
    
    # Confidence Metrics arguments
    confidence_group = moe_validation_parser.add_argument_group('Confidence Metrics')
    confidence_group.add_argument(
        "--enable-confidence",
        action="store_true",
        help="Enable confidence metrics for predictions that account for drift"
    )
    confidence_group.add_argument(
        "--drift-weight",
        type=float,
        default=0.5,
        help="Weight of drift impact in confidence calculation (0-1)"
    )
    confidence_group.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7, 0.9],
        help="Confidence thresholds for reporting (space-separated values between 0-1)"
    )
    
    # Add SATzilla training arguments
    satzilla_train_parser.add_argument(
        "--dimensions", "-d",
        type=int,
        default=2,
        help="Number of dimensions for benchmark functions"
    )
    
    satzilla_train_parser.add_argument(
        "--max-evaluations", "-e",
        type=int,
        default=1000,
        help="Maximum number of function evaluations per algorithm"
    )
    
    satzilla_train_parser.add_argument(
        "--functions", "-f",
        nargs="+",
        default=["sphere", "rosenbrock"],
        help="Benchmark functions to use for training"
    )
    
    satzilla_train_parser.add_argument(
        "--all-functions",
        action="store_true",
        help="Use all available benchmark functions for training"
    )
    
    satzilla_train_parser.add_argument(
        "--output-file", "-o",
        required=True,
        help="Output file path to save the trained selector"
    )
    
    # Add meta-learning command
    meta_parser = subparsers.add_parser(
        "meta",
        help="Run meta-learning"
    )
    
    # Add dynamic optimization command
    dynamic_parser = subparsers.add_parser(
        "dynamic_optimization",
        help="Run dynamic optimization visualization"
    )
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Set appropriate log level based on verbosity
    if hasattr(parsed_args, 'quiet') and parsed_args.quiet:
        parsed_args.log_level = 'ERROR'
    elif hasattr(parsed_args, 'verbose'):
        if parsed_args.verbose == 0:
            parsed_args.log_level = 'INFO'
        elif parsed_args.verbose == 1:
            parsed_args.log_level = 'DEBUG'
        else:
            parsed_args.log_level = 'DEBUG'  # Even more detailed debugging
    else:
        # Default log level if attributes are missing
        parsed_args.log_level = 'INFO'
    
    # Convert Namespace to dictionary
    return vars(parsed_args)
