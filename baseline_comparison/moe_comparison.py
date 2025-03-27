"""
MoE-enabled Baseline Comparison Framework

This module extends the baseline comparison framework to include the
Mixture of Experts (MoE) framework as a selection approach.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

# Import baseline comparison components
from baseline_comparison.comparison_runner import BaselineComparison

# Import MoE components
from baseline_comparison.moe_adapter import MoEBaselineAdapter

logger = logging.getLogger(__name__)


class MoEBaselineComparison(BaselineComparison):
    """
    Extended baseline comparison framework that includes MoE as a selection approach.
    
    This class adds the MoE framework to the standard baseline comparison, allowing
    direct performance comparison between MoE and other algorithm selection approaches.
    """
    
    def __init__(
        self,
        simple_baseline,
        meta_learner,
        enhanced_meta,
        satzilla_selector,
        moe_adapter: Optional[MoEBaselineAdapter] = None,
        moe_config: Optional[Dict[str, Any]] = None,
        max_evaluations: int = 10000,
        num_trials: int = 10,
        verbose: bool = True,
        output_dir: str = "results/baseline_comparison",
        model_path: str = None,
        moe_model_path: str = None
    ):
        """
        Initialize the extended comparison framework with MoE support.
        
        Args:
            simple_baseline: The simple baseline selector (random selection)
            meta_learner: The basic meta-learner
            enhanced_meta: The enhanced meta-optimizer with feature extraction
            satzilla_selector: The SATzilla-inspired selector
            moe_adapter: Optional pre-configured MoE adapter
            moe_config: Configuration for MoE if adapter not provided
            max_evaluations: Maximum number of function evaluations per algorithm
            num_trials: Number of trials to run per algorithm
            verbose: Whether to print progress information
            output_dir: Directory to save results and visualizations
            model_path: Optional path to trained model for SATzilla selector
            moe_model_path: Optional path to trained model for MoE
        """
        # Initialize base class
        super().__init__(
            simple_baseline=simple_baseline,
            meta_learner=meta_learner,
            enhanced_meta=enhanced_meta,
            satzilla_selector=satzilla_selector,
            max_evaluations=max_evaluations,
            num_trials=num_trials,
            verbose=verbose,
            output_dir=output_dir,
            model_path=model_path
        )
        
        # Initialize MoE adapter
        if moe_adapter is None and moe_config is not None:
            self.moe_adapter = MoEBaselineAdapter(
                config=moe_config,
                model_path=moe_model_path,
                verbose=verbose
            )
        else:
            self.moe_adapter = moe_adapter
        
        # Update the optimizer dictionary to include MoE
        if self.moe_adapter:
            self.optimizers = {
                "simple": self.simple_baseline,
                "meta": self.meta_learner,
                "enhanced": self.enhanced_meta,
                "satzilla": self.satzilla_selector,
                "moe": self.moe_adapter
            }
            
            # Update results structure to include MoE
            self.results["moe"] = {"best_fitness": [], "evaluations": [], "time": [], "convergence": [], "selections": []}
            
            # Make MoE use the same algorithm pool if possible
            if hasattr(self.moe_adapter, 'set_available_algorithms'):
                self.moe_adapter.set_available_algorithms(list(self.available_algorithms))
                logger.info("Set algorithms for MoE")
        
        if self.verbose:
            logger.info("Initialized MoEBaselineComparison with MoE adapter")
    
    def run_comparison(self, problem_name, problem_func, dimensions, max_evaluations, num_trials):
        """
        Run comparison between all approaches including MoE.
        
        Args:
            problem_name: Name of the problem
            problem_func: Problem function or generator
            dimensions: Dimensionality of the problem
            max_evaluations: Maximum number of evaluations
            num_trials: Number of trials to run
            
        Returns:
            Dictionary with comparison results
        """
        # Call parent method to run comparison for base optimizers
        results = super().run_comparison(
            problem_name=problem_name,
            problem_func=problem_func,
            dimensions=dimensions,
            max_evaluations=max_evaluations,
            num_trials=num_trials
        )
        
        # If we don't have an MoE adapter, return base results
        if not hasattr(self, 'moe_adapter') or self.moe_adapter is None:
            return results
        
        # Return the enhanced results
        return results
    
    def prepare_dataset_for_moe(self, problem_instance, target_column='target'):
        """
        Convert a problem instance to a format suitable for MoE.
        
        Args:
            problem_instance: Problem instance to convert
            target_column: Name of the target column
            
        Returns:
            DataFrame suitable for MoE
        """
        if isinstance(problem_instance, pd.DataFrame):
            return problem_instance
        
        # Try to convert to DataFrame
        try:
            # Extract features
            if hasattr(problem_instance, 'get_features'):
                features = problem_instance.get_features()
                df = pd.DataFrame(features)
                
                # Add target if available
                if hasattr(problem_instance, 'get_target'):
                    target = problem_instance.get_target()
                    df[target_column] = target
                
                return df
            elif hasattr(problem_instance, 'to_dataframe'):
                return problem_instance.to_dataframe()
        except (AttributeError, TypeError) as e:
            logger.error(f"Cannot convert problem instance to DataFrame: {e}")
        
        # Fallback: create a simple DataFrame with the raw data
        if hasattr(problem_instance, '__dict__'):
            return pd.DataFrame([problem_instance.__dict__])
        
        # Last resort: try to convert directly
        try:
            return pd.DataFrame([problem_instance])
        except:
            logger.error("Failed to convert problem instance to DataFrame")
            return pd.DataFrame()
    
    def cross_validate_all(self, X, y, n_splits=5, method='patient_aware'):
        """
        Perform cross-validation for all selectors including MoE.
        
        Args:
            X: Features DataFrame
            y: Target values Series
            n_splits: Number of CV splits
            method: Validation method for time series
            
        Returns:
            Dictionary with validation results for all selectors
        """
        results = {}
        
        # Cross-validate MoE if available
        if hasattr(self, 'moe_adapter') and self.moe_adapter is not None:
            try:
                moe_scores = self.moe_adapter.cross_validate(X, y, n_splits=n_splits, method=method)
                results['moe'] = moe_scores
                logger.info("MoE cross-validation completed")
            except Exception as e:
                logger.error(f"Error in MoE cross-validation: {e}")
        
        # Add cross-validation for other selectors if they support it
        for name, selector in [
            ('simple', self.simple_baseline),
            ('meta', self.meta_learner),
            ('enhanced', self.enhanced_meta),
            ('satzilla', self.satzilla_selector)
        ]:
            if hasattr(selector, 'cross_validate'):
                try:
                    scores = selector.cross_validate(X, y, n_splits=n_splits)
                    results[name] = scores
                    logger.info(f"{name} cross-validation completed")
                except Exception as e:
                    logger.error(f"Error in {name} cross-validation: {e}")
        
        return results
    
    def get_summary_with_moe(self):
        """
        Get a summary of results including MoE.
        
        Returns:
            DataFrame with performance metrics for all selectors
        """
        # Get the base summary (will exclude MoE)
        summary = self.get_summary_dataframe()
        
        # If we have MoE results, add them
        if 'moe' in self.results and self.results['moe']['best_fitness']:
            moe_data = {
                'Selector': 'MoE',
                'Best Fitness (mean)': np.mean(self.results['moe']['best_fitness']),
                'Best Fitness (std)': np.std(self.results['moe']['best_fitness']),
                'Evaluations (mean)': np.mean(self.results['moe']['evaluations']),
                'Time (mean)': np.mean(self.results['moe']['time']),
                'Success Rate': self._calculate_success_rate(self.results['moe']['best_fitness']),
                'Overall Rank': 0  # Will be calculated later
            }
            
            # Convert to DataFrame and append
            moe_df = pd.DataFrame([moe_data])
            summary = pd.concat([summary, moe_df], ignore_index=True)
            
            # Recalculate ranks
            summary['Overall Rank'] = summary['Best Fitness (mean)'].rank()
        
        return summary
    
    def run_supervised_comparison(self, X_train, y_train, X_test, y_test, **kwargs):
        """
        Run a supervised learning comparison between all approaches including MoE.
        
        Args:
            X_train: Training features DataFrame
            y_train: Training target Series
            X_test: Testing features DataFrame
            y_test: Testing target Series
            **kwargs: Additional arguments to pass to trainers/predictors
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        # Train and test the simple baseline if available
        if hasattr(self, 'simple_baseline') and self.simple_baseline and hasattr(self.simple_baseline, 'train'):
            self.simple_baseline.train(X_train, y_train)
            simple_preds = self.simple_baseline.predict(X_test)
            results['simple_baseline'] = self.compute_metrics(y_test, simple_preds)
        
        # Train and test the meta-learner baseline if available
        if hasattr(self, 'meta_learner') and self.meta_learner and hasattr(self.meta_learner, 'train'):
            self.meta_learner.train(X_train, y_train)
            meta_preds = self.meta_learner.predict(X_test)
            results['meta_learner'] = self.compute_metrics(y_test, meta_preds)
        
        # Train and test the enhanced meta baseline if available
        if hasattr(self, 'enhanced_meta') and self.enhanced_meta and hasattr(self.enhanced_meta, 'train'):
            self.enhanced_meta.train(X_train, y_train)
            enhanced_preds = self.enhanced_meta.predict(X_test)
            results['enhanced_meta'] = self.compute_metrics(y_test, enhanced_preds)
        
        # Train and test the SATzilla-style selector baseline if available
        if hasattr(self, 'satzilla_selector') and self.satzilla_selector and hasattr(self.satzilla_selector, 'train'):
            self.satzilla_selector.train(X_train, y_train)
            satzilla_preds = self.satzilla_selector.predict(X_test)
            results['satzilla_selector'] = self.compute_metrics(y_test, satzilla_preds)
        
        # Train and test the MoE adapter if available
        if hasattr(self, 'moe_adapter') and self.moe_adapter:
            self.moe_adapter.train(X_train, y_train)
            # Use the adapter's predict method with ground truth for tracking metrics
            moe_preds = self.moe_adapter.predict(X_test, y=y_test)
            
            # Compute standard metrics for comparison with other approaches
            standard_metrics = self.compute_metrics(y_test, moe_preds)
            
            # Compute MoE-specific comprehensive metrics
            moe_metrics = self.moe_adapter.compute_metrics("comparison_run")
            
            # Combine standard and MoE-specific metrics
            results['moe'] = {
                'standard': standard_metrics,
                'moe_specific': moe_metrics
            }
        
        return results
    
    def compute_metrics(self, actual, predicted):
        """
        Compute evaluation metrics for a set of predictions.
        
        Args:
            actual: Actual values (ground truth)
            predicted: Predicted values
            
        Returns:
            Dictionary with computed metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Convert inputs to numpy arrays
        if not isinstance(actual, np.ndarray):
            actual = np.array(actual)
        if not isinstance(predicted, np.ndarray):
            predicted = np.array(predicted)
            
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        # Calculate percentage error metrics
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def visualize_supervised_comparison(self, results, output_dir=None, prefix="comparison"):
        """
        Visualize comparison results from supervised learning.
        
        Args:
            results: Results dictionary from run_supervised_comparison
            output_dir: Directory to save visualizations
            prefix: Prefix for output files
            
        Returns:
            DataFrame with comparison metrics
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import json
        
        # Use instance output_dir if none provided
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a DataFrame for standard metrics
        metrics_df = pd.DataFrame()
        
        for approach, metrics in results.items():
            # Handle MoE's nested metrics structure
            if approach == 'moe' and isinstance(metrics, dict) and 'standard' in metrics:
                approach_metrics = metrics['standard'].copy()
            else:
                approach_metrics = metrics.copy() if isinstance(metrics, dict) else {}
                
            approach_metrics['approach'] = approach
            metrics_df = pd.concat([metrics_df, pd.DataFrame([approach_metrics])], ignore_index=True)
        
        # Plot RMSE comparison
        if 'rmse' in metrics_df.columns:
            plt.figure(figsize=(10, 6))
            bar = plt.bar(metrics_df['approach'], metrics_df['rmse'])
            plt.title('RMSE Comparison')
            plt.xlabel('Approach')
            plt.ylabel('RMSE')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Add values on top of bars
            for rect in bar:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0)
                
            plt.savefig(f"{output_dir}/{prefix}_rmse_comparison.png")
            
        # Plot MAE comparison
        if 'mae' in metrics_df.columns:
            plt.figure(figsize=(10, 6))
            bar = plt.bar(metrics_df['approach'], metrics_df['mae'])
            plt.title('MAE Comparison')
            plt.xlabel('Approach')
            plt.ylabel('MAE')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Add values on top of bars
            for rect in bar:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0)
                
            plt.savefig(f"{output_dir}/{prefix}_mae_comparison.png")
        
        # Save the raw metrics to CSV
        metrics_df.to_csv(f"{output_dir}/{prefix}_metrics.csv", index=False)
        
        # Save the complete results data including MoE-specific metrics
        with open(f"{output_dir}/{prefix}_full_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: str(x) if isinstance(x, (np.ndarray, np.number)) else x)
        
        # Create MoE-specific visualization if available
        if 'moe' in results and isinstance(results['moe'], dict) and 'moe_specific' in results['moe']:
            moe_metrics = results['moe']['moe_specific']
            
            # Plot expert contributions if available
            if 'expert_contribution' in moe_metrics and 'expert_dominance_percentage' in moe_metrics['expert_contribution']:
                expert_usage = moe_metrics['expert_contribution']['expert_dominance_percentage']
                if expert_usage:
                    plt.figure(figsize=(10, 6))
                    plt.pie(
                        list(expert_usage.values()), 
                        labels=list(expert_usage.keys()),
                        autopct='%1.1f%%',
                        startangle=90
                    )
                    plt.title('Expert Usage Distribution')
                    plt.savefig(f"{output_dir}/{prefix}_expert_usage.png")
            
            # Plot confidence distribution if available
            if 'confidence' in moe_metrics and 'confidence_histogram' in moe_metrics['confidence']:
                confidence_hist = moe_metrics['confidence']['confidence_histogram']
                if confidence_hist and 'bins' in confidence_hist and 'values' in confidence_hist:
                    plt.figure(figsize=(10, 6))
                    plt.bar(confidence_hist['bins'][:-1], confidence_hist['values'], 
                           width=confidence_hist['bins'][1]-confidence_hist['bins'][0])
                    plt.title('Confidence Score Distribution')
                    plt.xlabel('Confidence')
                    plt.ylabel('Frequency')
                    plt.savefig(f"{output_dir}/{prefix}_confidence_distribution.png")
                    plt.close()
        
        return metrics_df
        
    def visualize_results(self, include_moe=True, save=True):
        """
        Generate visualizations for the baseline comparison results.
        
        Args:
            include_moe: Whether to include MoE in visualizations
            save: Whether to save visualizations to disk
            
        Returns:
            Dictionary of visualization paths
        """
        # Check that we have results to visualize
        if not self.results:
            logger.warning("No results to visualize")
            return {}
        
        # Initialize result container
        vis_paths = {}
        
        # Create visualizer if needed
        if not hasattr(self, 'visualizer') or self.visualizer is None:
            from baseline_comparison.visualization import ComparisonVisualizer
            self.visualizer = ComparisonVisualizer(self.results, self.output_dir)
        
        # Determine which optimizers to include
        optimizers = ["simple", "meta", "enhanced", "satzilla"]
        if include_moe and "moe" in self.results and self.moe_adapter is not None:
            optimizers.append("moe")
            
        # Run standard visualizations
        try:
            # Performance comparison
            vis_paths["performance"] = self.visualizer.performance_comparison(save=save)
            
            # Selections frequency
            vis_paths["selections"] = self.visualizer.selection_frequency(save=save)
            
            # Convergence curve
            vis_paths["convergence"] = self.visualizer.convergence_curve(save=save)
            
            # Radar chart
            vis_paths["radar"] = self.visualizer.radar_chart(optimizers=optimizers, save=save)
            
            # Boxplot
            vis_paths["boxplot"] = self.visualizer.boxplot(optimizers=optimizers, save=save)
            
            # Heatmap
            vis_paths["heatmap"] = self.visualizer.heatmap(save=save)
            
            if include_moe and "moe" in self.results and self.moe_adapter is not None:
                # Generate MoE-specific visualizations if MoE adapter has metrics available
                if hasattr(self.moe_adapter, 'get_metrics') and callable(getattr(self.moe_adapter, 'get_metrics')):
                    try:
                        moe_metrics = self.moe_adapter.get_metrics()
                        if moe_metrics:
                            # Get metrics calculator from MoE adapter if possible
                            metrics_calculator = None
                            if hasattr(self.moe_adapter, 'metrics_calculator'):
                                metrics_calculator = self.moe_adapter.metrics_calculator
                            else:
                                # Create a new metrics calculator if one doesn't exist
                                from baseline_comparison.moe_metrics import MoEMetricsCalculator
                                metrics_calculator = MoEMetricsCalculator(output_dir=self.output_dir)
                            
                            # Generate MoE-specific visualizations
                            if metrics_calculator:
                                vis_paths["moe_metrics"] = metrics_calculator.visualize_metrics(moe_metrics)
                                
                                # Create comparative visualizations between MoE and baselines
                                # Prepare baseline metrics in compatible format
                                baseline_metrics_dict = {}
                                for optimizer_name in [opt for opt in optimizers if opt != "moe"]:
                                    # Extract relevant metrics for comparison
                                    if optimizer_name in self.results:
                                        # Get relevant performance metrics
                                        baseline_metrics = {}
                                        
                                        # Include standard performance metrics if available
                                        for metric in ['rmse', 'mae', 'r2']:
                                            if metric in self.results[optimizer_name]:
                                                baseline_metrics[metric] = self.results[optimizer_name][metric]
                                        
                                        # Add training and inference time if available
                                        if 'time' in self.results[optimizer_name]:
                                            baseline_metrics['training_time'] = np.mean(self.results[optimizer_name]['time'])
                                        
                                        # Add algorithm selections if available
                                        if 'selections' in self.results[optimizer_name]:
                                            baseline_metrics['selections'] = self.results[optimizer_name]['selections']
                                            
                                        if baseline_metrics:  # Only add if we have some metrics
                                            baseline_metrics_dict[optimizer_name] = baseline_metrics
                                
                                # Generate comparison visualizations if we have baselines to compare with
                                if baseline_metrics_dict:
                                    vis_paths["moe_comparison"] = metrics_calculator.visualize_comparison(
                                        moe_metrics=moe_metrics,
                                        baseline_metrics=baseline_metrics_dict,
                                        name="moe_baseline_comparison"
                                    )
                    except Exception as e:
                        logger.error(f"Error generating MoE visualizations: {e}")
                        import traceback
                        traceback.print_exc()
                
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            
        return vis_paths


# Helper function to create a configured MoE adapter
def create_moe_adapter(config_path=None, config=None, model_path=None, verbose=False):
    """
    Create a configured MoE adapter for use in baseline comparison.
    
    Args:
        config_path: Path to configuration file
        config: Configuration dictionary (used if config_path is None)
        model_path: Path to trained model
        verbose: Whether to display detailed logs
        
    Returns:
        Configured MoEBaselineAdapter instance
    """
    # Load configuration from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded MoE configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading MoE configuration from {config_path}: {e}")
            config = {}
    
    # Create and return the adapter
    try:
        adapter = MoEBaselineAdapter(
            config=config,
            model_path=model_path,
            verbose=verbose
        )
        return adapter
    except Exception as e:
        logger.error(f"Error creating MoE adapter: {e}")
        return None


def visualize_moe_with_baselines(moe_metrics, baseline_metrics, output_dir="results/baseline_comparison"):
    """
    Create visualizations comparing MoE with baseline approaches.
    
    This is a standalone utility function that can be used outside of the MoEBaselineComparison class.
    
    Args:
        moe_metrics: Dictionary of MoE metrics
        baseline_metrics: Dictionary mapping baseline names to their metrics
        output_dir: Directory to save visualization files
        
    Returns:
        List of paths to generated visualization files
    """
    from baseline_comparison.moe_metrics import MoEMetricsCalculator
    import os
    from pathlib import Path
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create metrics calculator
    metrics_calculator = MoEMetricsCalculator(output_dir=output_path)
    
    # Generate comparison visualizations
    try:
        paths = metrics_calculator.visualize_comparison(
            moe_metrics=moe_metrics,
            baseline_metrics=baseline_metrics,
            name="moe_baseline_comparison"
        )
        return paths
    except Exception as e:
        logger.error(f"Error generating comparison visualizations: {e}")
        import traceback
        traceback.print_exc()
        return []
