"""
Performance Benchmarking Framework

This module provides tools for benchmarking the MoE framework across different
configurations, measuring performance metrics such as prediction accuracy,
execution time, and memory usage.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import psutil
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from moe_framework.workflow.moe_pipeline import MoEPipeline
from moe_framework.integration.integration_layer import (
    IntegrationLayer, 
    WeightedAverageIntegration,
    AdaptiveIntegration
)
from moe_framework.gating.gating_network import (
    GatingNetwork,
    QualityAwareWeighting,
    MetaLearnerGating
)

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runner for executing benchmarks across different MoE configurations.
    
    This class provides methods for measuring performance metrics across
    different configurations of the MoE framework, including:
    - Prediction accuracy
    - Execution time
    - Memory usage
    - Scalability
    """
    
    def __init__(self, 
                output_dir: str = None,
                create_visualizations: bool = True):
        """
        Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
            create_visualizations: Whether to create visualization charts
        """
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'benchmark_results'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.create_visualizations = create_visualizations
        self.results = []
        
        logger.info(f"Initialized BenchmarkRunner with output_dir={self.output_dir}")
    
    def benchmark_pipeline_configuration(self,
                                       config: Dict[str, Any],
                                       data: pd.DataFrame,
                                       features: List[str],
                                       target: str,
                                       test_data: Optional[pd.DataFrame] = None,
                                       name: str = None,
                                       iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark a specific pipeline configuration.
        
        Args:
            config: Pipeline configuration
            data: Training data
            features: Feature columns
            target: Target column
            test_data: Optional test data (if None, will use 20% of training data)
            name: Name for this benchmark
            iterations: Number of iterations to run for averaging results
            
        Returns:
            Dictionary with benchmark results
        """
        name = name or f"config_{len(self.results) + 1}"
        
        # Split data if test_data not provided
        if test_data is None:
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        else:
            train_data = data
        
        # Prepare result dictionary
        result = {
            'name': name,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'features': len(features),
            },
            'performance': {
                'accuracy': [],
                'execution_time': [],
                'memory_usage': [],
            },
            'iterations': iterations
        }
        
        # Run multiple iterations
        for i in range(iterations):
            logger.info(f"Running benchmark '{name}' - Iteration {i+1}/{iterations}")
            
            # Track memory and time
            start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
            start_time = time.time()
            
            # Initialize and train pipeline
            pipeline = MoEPipeline(config=config)
            
            try:
                # Train the pipeline
                pipeline.train(train_data, target_column=target)
                
                # Measure post-training memory
                post_train_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                post_train_time = time.time()
                
                # Make predictions
                X_test = test_data[features]
                y_true = test_data[target].values
                
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics = {
                    'mse': mean_squared_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred)
                }
                
                # Final measurements
                end_time = time.time()
                end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                
                # Record performance metrics
                result['performance']['accuracy'].append(metrics)
                result['performance']['execution_time'].append({
                    'total': end_time - start_time,
                    'training': post_train_time - start_time,
                    'prediction': end_time - post_train_time
                })
                result['performance']['memory_usage'].append({
                    'start': start_mem,
                    'post_training': post_train_mem,
                    'end': end_mem,
                    'delta': end_mem - start_mem
                })
                
                logger.info(f"Iteration {i+1} completed - "
                           f"RMSE: {metrics['rmse']:.4f}, "
                           f"Time: {end_time - start_time:.2f}s, "
                           f"Memory: {end_mem - start_mem:.2f}MB")
                
            except Exception as e:
                logger.error(f"Error in benchmark '{name}' iteration {i+1}: {str(e)}")
                # Record error but continue with next iteration
                result['performance']['errors'] = result.get('performance', {}).get('errors', []) + [str(e)]
        
        # Calculate averages
        self._calculate_averages(result)
        
        # Store result
        self.results.append(result)
        
        # Save result to disk
        self._save_result(result)
        
        return result
    
    def benchmark_integration_strategies(self,
                                        data: pd.DataFrame,
                                        features: List[str],
                                        target: str,
                                        base_config: Dict[str, Any],
                                        test_data: Optional[pd.DataFrame] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Benchmark different integration strategies.
        
        Args:
            data: Training data
            features: Feature columns
            target: Target column
            base_config: Base pipeline configuration
            test_data: Optional test data
            
        Returns:
            Dictionary mapping strategy names to benchmark results
        """
        strategies = {
            'weighted_average': {'strategy': 'weighted_average'},
            'confidence_based': {
                'strategy': 'adaptive',
                'confidence_threshold': 0.7,
                'min_experts': 1
            },
            'quality_aware': {
                'strategy': 'adaptive',
                'quality_threshold': 0.75,
                'quality_impact': 0.8
            },
            'advanced_fusion': {
                'strategy': 'adaptive',
                'confidence_threshold': 0.7,
                'quality_threshold': 0.75,
                'min_experts': 1,
                'quality_impact': 0.8
            }
        }
        
        results = {}
        
        # Benchmark each strategy
        for strategy_name, strategy_config in strategies.items():
            # Create configuration with this strategy
            config = base_config.copy()
            config['integration'] = strategy_config
            
            # Run benchmark
            result = self.benchmark_pipeline_configuration(
                config=config,
                data=data,
                features=features,
                target=target,
                test_data=test_data,
                name=f"integration_{strategy_name}"
            )
            
            results[strategy_name] = result
        
        # Create comparison visualization
        if self.create_visualizations:
            self._create_strategy_comparison_charts(
                results=list(results.values()),
                chart_name="integration_strategies_comparison"
            )
        
        return results
    
    def benchmark_gating_networks(self,
                                data: pd.DataFrame,
                                features: List[str],
                                target: str,
                                base_config: Dict[str, Any],
                                test_data: Optional[pd.DataFrame] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Benchmark different gating network configurations.
        
        Args:
            data: Training data
            features: Feature columns
            target: Target column
            base_config: Base pipeline configuration
            test_data: Optional test data
            
        Returns:
            Dictionary mapping gating network names to benchmark results
        """
        gating_configs = {
            'fixed_weights': {
                'type': 'fixed',
                'weights': {
                    'physiological_expert': 0.4,
                    'behavioral_expert': 0.3,
                    'environmental_expert': 0.2,
                    'medication_history_expert': 0.1
                }
            },
            'quality_aware': {
                'type': 'quality_aware',
                'quality_threshold': 0.7,
                'default_weights': {
                    'physiological_expert': 0.4,
                    'behavioral_expert': 0.3,
                    'environmental_expert': 0.2,
                    'medication_history_expert': 0.1
                }
            },
            'meta_learner': {
                'type': 'meta_learner',
                'learning_rate': 0.01,
                'regularization': 0.001,
                'context_features': ['data_quality']
            }
        }
        
        results = {}
        
        # Benchmark each gating configuration
        for gating_name, gating_config in gating_configs.items():
            # Create configuration with this gating network
            config = base_config.copy()
            config['gating_network'] = gating_config
            
            # Run benchmark
            result = self.benchmark_pipeline_configuration(
                config=config,
                data=data,
                features=features,
                target=target,
                test_data=test_data,
                name=f"gating_{gating_name}"
            )
            
            results[gating_name] = result
        
        # Create comparison visualization
        if self.create_visualizations:
            self._create_strategy_comparison_charts(
                results=list(results.values()),
                chart_name="gating_network_comparison"
            )
        
        return results
    
    def benchmark_scaling(self,
                        data_generator: Callable[[int], pd.DataFrame],
                        features: List[str],
                        target: str,
                        base_config: Dict[str, Any],
                        sample_sizes: List[int]) -> Dict[str, Any]:
        """
        Benchmark how the pipeline scales with increasing data size.
        
        Args:
            data_generator: Function that generates dataframes of specified size
            features: Feature columns
            target: Target column
            base_config: Base pipeline configuration
            sample_sizes: List of sample sizes to test
            
        Returns:
            Dictionary with scaling benchmark results
        """
        scaling_results = {
            'name': 'scaling_benchmark',
            'timestamp': datetime.now().isoformat(),
            'sample_sizes': sample_sizes,
            'metrics': {
                'training_time': [],
                'prediction_time': [],
                'memory_usage': [],
                'accuracy': []
            }
        }
        
        for size in sample_sizes:
            logger.info(f"Running scaling benchmark with {size} samples")
            
            # Generate data of this size
            data = data_generator(size)
            
            # Split into train and test
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            
            # Run benchmark
            result = self.benchmark_pipeline_configuration(
                config=base_config,
                data=train_data,
                features=features,
                target=target,
                test_data=test_data,
                name=f"scaling_{size}_samples",
                iterations=1  # One iteration is enough for scaling tests
            )
            
            # Extract metrics
            scaling_results['metrics']['training_time'].append(
                result['performance']['average']['execution_time']['training']
            )
            scaling_results['metrics']['prediction_time'].append(
                result['performance']['average']['execution_time']['prediction']
            )
            scaling_results['metrics']['memory_usage'].append(
                result['performance']['average']['memory_usage']['delta']
            )
            scaling_results['metrics']['accuracy'].append(
                result['performance']['average']['accuracy']['rmse']
            )
        
        # Create scaling visualization
        if self.create_visualizations:
            self._create_scaling_charts(scaling_results)
        
        # Save results
        self._save_result(scaling_results, filename="scaling_benchmark.json")
        
        return scaling_results
    
    def compare_all_results(self) -> Dict[str, Any]:
        """
        Compare all benchmark results and generate a comprehensive report.
        
        Returns:
            Dictionary with comparison results
        """
        if not self.results:
            logger.warning("No benchmark results available for comparison")
            return {}
        
        # Create comparison structure
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': len(self.results),
            'configurations': [r['name'] for r in self.results],
            'accuracy': {
                'rmse': [r['performance']['average']['accuracy']['rmse'] for r in self.results],
                'r2': [r['performance']['average']['accuracy']['r2'] for r in self.results]
            },
            'execution_time': {
                'total': [r['performance']['average']['execution_time']['total'] for r in self.results],
                'training': [r['performance']['average']['execution_time']['training'] for r in self.results],
                'prediction': [r['performance']['average']['execution_time']['prediction'] for r in self.results]
            },
            'memory_usage': {
                'delta': [r['performance']['average']['memory_usage']['delta'] for r in self.results]
            }
        }
        
        # Create comparison visualizations
        if self.create_visualizations:
            self._create_comparison_charts(comparison)
        
        # Save comparison
        comparison_path = os.path.join(self.output_dir, "comparison_summary.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Saved comparison summary to {comparison_path}")
        
        return comparison
    
    def _calculate_averages(self, result: Dict[str, Any]) -> None:
        """Calculate average metrics across iterations."""
        if 'performance' not in result:
            return
        
        perf = result['performance']
        result['performance']['average'] = {
            'accuracy': {},
            'execution_time': {},
            'memory_usage': {}
        }
        
        # Calculate average accuracy metrics
        if 'accuracy' in perf and perf['accuracy']:
            for metric in perf['accuracy'][0].keys():
                values = [run[metric] for run in perf['accuracy']]
                result['performance']['average']['accuracy'][metric] = np.mean(values)
        
        # Calculate average execution time
        if 'execution_time' in perf and perf['execution_time']:
            for key in perf['execution_time'][0].keys():
                values = [run[key] for run in perf['execution_time']]
                result['performance']['average']['execution_time'][key] = np.mean(values)
        
        # Calculate average memory usage
        if 'memory_usage' in perf and perf['memory_usage']:
            for key in perf['memory_usage'][0].keys():
                values = [run[key] for run in perf['memory_usage']]
                result['performance']['average']['memory_usage'][key] = np.mean(values)
    
    def _save_result(self, result: Dict[str, Any], filename: str = None) -> None:
        """Save benchmark result to disk."""
        if filename is None:
            filename = f"{result['name']}_{int(time.time())}.json"
        
        result_path = os.path.join(self.output_dir, filename)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved benchmark result to {result_path}")
    
    def _create_strategy_comparison_charts(self, results: List[Dict[str, Any]], chart_name: str) -> None:
        """Create charts comparing different strategies."""
        if not results:
            return
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract names and metrics
        names = [r['name'] for r in results]
        rmse = [r['performance']['average']['accuracy']['rmse'] for r in results]
        r2 = [r['performance']['average']['accuracy']['r2'] for r in results]
        train_time = [r['performance']['average']['execution_time']['training'] for r in results]
        memory = [r['performance']['average']['memory_usage']['delta'] for r in results]
        
        # Plot RMSE (lower is better)
        axs[0, 0].bar(names, rmse)
        axs[0, 0].set_title('RMSE (lower is better)')
        axs[0, 0].set_ylabel('RMSE')
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot R² (higher is better)
        axs[0, 1].bar(names, r2)
        axs[0, 1].set_title('R² Score (higher is better)')
        axs[0, 1].set_ylabel('R²')
        axs[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot training time (lower is better)
        axs[1, 0].bar(names, train_time)
        axs[1, 0].set_title('Training Time (lower is better)')
        axs[1, 0].set_ylabel('Seconds')
        axs[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot memory usage (lower is better)
        axs[1, 1].bar(names, memory)
        axs[1, 1].set_title('Memory Usage (lower is better)')
        axs[1, 1].set_ylabel('MB')
        axs[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        chart_path = os.path.join(self.output_dir, f"{chart_name}.png")
        plt.savefig(chart_path)
        logger.info(f"Saved comparison chart to {chart_path}")
        plt.close()
    
    def _create_scaling_charts(self, scaling_results: Dict[str, Any]) -> None:
        """Create charts showing scaling behavior."""
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        sample_sizes = scaling_results['sample_sizes']
        
        # Plot training time scaling
        axs[0, 0].plot(sample_sizes, scaling_results['metrics']['training_time'], 'o-')
        axs[0, 0].set_title('Training Time Scaling')
        axs[0, 0].set_xlabel('Number of Samples')
        axs[0, 0].set_ylabel('Time (seconds)')
        axs[0, 0].grid(True)
        
        # Plot prediction time scaling
        axs[0, 1].plot(sample_sizes, scaling_results['metrics']['prediction_time'], 'o-')
        axs[0, 1].set_title('Prediction Time Scaling')
        axs[0, 1].set_xlabel('Number of Samples')
        axs[0, 1].set_ylabel('Time (seconds)')
        axs[0, 1].grid(True)
        
        # Plot memory usage scaling
        axs[1, 0].plot(sample_sizes, scaling_results['metrics']['memory_usage'], 'o-')
        axs[1, 0].set_title('Memory Usage Scaling')
        axs[1, 0].set_xlabel('Number of Samples')
        axs[1, 0].set_ylabel('Memory (MB)')
        axs[1, 0].grid(True)
        
        # Plot accuracy scaling
        axs[1, 1].plot(sample_sizes, scaling_results['metrics']['accuracy'], 'o-')
        axs[1, 1].set_title('RMSE Scaling (lower is better)')
        axs[1, 1].set_xlabel('Number of Samples')
        axs[1, 1].set_ylabel('RMSE')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        chart_path = os.path.join(self.output_dir, "scaling_benchmark.png")
        plt.savefig(chart_path)
        logger.info(f"Saved scaling chart to {chart_path}")
        plt.close()
    
    def _create_comparison_charts(self, comparison: Dict[str, Any]) -> None:
        """Create comparison charts across all benchmarks."""
        names = comparison['configurations']
        
        # Create accuracy comparison
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        axs[0].bar(names, comparison['accuracy']['rmse'])
        axs[0].set_title('RMSE Comparison (lower is better)')
        axs[0].set_ylabel('RMSE')
        axs[0].tick_params(axis='x', rotation=45)
        
        axs[1].bar(names, comparison['accuracy']['r2'])
        axs[1].set_title('R² Comparison (higher is better)')
        axs[1].set_ylabel('R²')
        axs[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "accuracy_comparison.png"))
        plt.close()
        
        # Create performance comparison
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        axs[0].bar(names, comparison['execution_time']['total'])
        axs[0].set_title('Total Execution Time (lower is better)')
        axs[0].set_ylabel('Seconds')
        axs[0].tick_params(axis='x', rotation=45)
        
        axs[1].bar(names, comparison['memory_usage']['delta'])
        axs[1].set_title('Memory Usage (lower is better)')
        axs[1].set_ylabel('MB')
        axs[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_comparison.png"))
        plt.close()


def run_standard_benchmark_suite(data_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a standard suite of benchmarks on the provided dataset.
    
    Args:
        data_path: Path to the dataset CSV
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary with benchmark results
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Define features and target
    features = [col for col in data.columns if col != 'target']
    target = 'target'
    
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir=output_dir)
    
    # Define base configuration
    base_config = {
        'experts': {
            'physiological_expert': {
                'type': 'physiological',
                'model': 'gradient_boosting'
            },
            'behavioral_expert': {
                'type': 'behavioral',
                'model': 'random_forest'
            },
            'environmental_expert': {
                'type': 'environmental',
                'model': 'linear'
            },
            'medication_history_expert': {
                'type': 'medication_history',
                'model': 'gradient_boosting'
            }
        },
        'gating_network': {
            'type': 'quality_aware',
            'quality_threshold': 0.7
        },
        'integration': {
            'strategy': 'weighted_average'
        }
    }
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Benchmark integration strategies
    integration_results = runner.benchmark_integration_strategies(
        data=train_data,
        features=features,
        target=target,
        base_config=base_config,
        test_data=test_data
    )
    
    # Benchmark gating networks
    gating_results = runner.benchmark_gating_networks(
        data=train_data,
        features=features,
        target=target,
        base_config=base_config,
        test_data=test_data
    )
    
    # Compare all results
    comparison = runner.compare_all_results()
    
    return {
        'integration': integration_results,
        'gating': gating_results,
        'comparison': comparison
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MoE performance benchmarks')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--output', type=str, default=None, help='Output directory for results')
    
    args = parser.parse_args()
    
    results = run_standard_benchmark_suite(args.data, args.output)
    print(f"Benchmark completed. Results saved to {args.output or 'benchmark_results/'}")
