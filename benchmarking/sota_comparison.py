"""
sota_comparison.py
----------------
Framework for comparing optimization algorithms against state-of-the-art variants.
Includes implementations of SOTA variants and comparison metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Callable
from dataclasses import dataclass
import pandas as pd
from scipy import stats
from .statistical_analysis import StatisticalAnalyzer

@dataclass
class ComparisonMetric:
    name: str
    better: str  # 'higher' or 'lower'
    compute: Callable
    description: str

@dataclass
class ComparisonResult:
    algorithm: str
    reference: str
    metrics: Dict[str, float]
    statistical_tests: Dict[str, Any]
    is_better: bool
    description: str

class SOTAComparison:
    def __init__(self):
        """Initialize SOTA comparison framework"""
        self.metrics = {
            'mean_fitness': ComparisonMetric(
                name='Mean Fitness',
                better='lower',
                compute=lambda x: np.mean(x),
                description='Average fitness value across all runs'
            ),
            'best_fitness': ComparisonMetric(
                name='Best Fitness',
                better='lower',
                compute=lambda x: np.min(x),
                description='Best fitness value found across all runs'
            ),
            'success_rate': ComparisonMetric(
                name='Success Rate',
                better='higher',
                compute=lambda x, threshold: np.mean(x <= threshold),
                description='Fraction of runs reaching the success threshold'
            ),
            'convergence_speed': ComparisonMetric(
                name='Convergence Speed',
                better='higher',
                compute=self._compute_convergence_speed,
                description='Rate of improvement in fitness values'
            ),
            'robustness': ComparisonMetric(
                name='Robustness',
                better='lower',
                compute=lambda x: np.std(x),
                description='Standard deviation of fitness values'
            )
        }
        
        self.statistical_analyzer = StatisticalAnalyzer()
    
    def compare_with_reference(
            self,
            algorithm_results: Dict[str, List[float]],
            reference_results: Dict[str, List[float]],
            success_threshold: float = None
        ) -> Dict[str, ComparisonResult]:
        """
        Compare algorithm results with reference implementation results.
        
        Args:
            algorithm_results: Dictionary mapping function names to lists of fitness values
            reference_results: Dictionary mapping function names to lists of reference fitness values
            success_threshold: Optional threshold for success rate calculation
            
        Returns:
            Dictionary mapping function names to ComparisonResult objects
        """
        results = {}
        
        for func_name in algorithm_results:
            if func_name not in reference_results:
                continue
                
            alg_fitness = np.array(algorithm_results[func_name])
            ref_fitness = np.array(reference_results[func_name])
            
            # Compute all metrics
            metrics = {}
            for metric_name, metric in self.metrics.items():
                if metric_name == 'success_rate' and success_threshold is not None:
                    alg_value = metric.compute(alg_fitness, success_threshold)
                    ref_value = metric.compute(ref_fitness, success_threshold)
                else:
                    alg_value = metric.compute(alg_fitness)
                    ref_value = metric.compute(ref_fitness)
                    
                metrics[metric_name] = {
                    'algorithm': alg_value,
                    'reference': ref_value,
                    'relative_improvement': self._compute_relative_improvement(
                        alg_value, ref_value, metric.better
                    )
                }
            
            # Perform statistical tests
            statistical_tests = {
                'mann_whitney': self.statistical_analyzer.mann_whitney_test(
                    alg_fitness, ref_fitness
                ),
                'effect_size': self.statistical_analyzer._cliff_delta(
                    alg_fitness, ref_fitness
                )
            }
            
            # Determine if algorithm is better overall
            is_better = self._is_better_overall(metrics)
            
            # Generate comparison description
            description = self._generate_comparison_description(
                metrics, statistical_tests, func_name
            )
            
            results[func_name] = ComparisonResult(
                algorithm='Algorithm',
                reference='Reference',
                metrics=metrics,
                statistical_tests=statistical_tests,
                is_better=is_better,
                description=description
            )
        
        return results
    
    def generate_comparison_report(
            self,
            comparison_results: Dict[str, ComparisonResult]
        ) -> pd.DataFrame:
        """
        Generate a detailed comparison report.
        
        Args:
            comparison_results: Dictionary of ComparisonResult objects
            
        Returns:
            DataFrame containing detailed comparison results
        """
        rows = []
        
        for func_name, result in comparison_results.items():
            for metric_name, values in result.metrics.items():
                rows.append({
                    'Function': func_name,
                    'Metric': metric_name,
                    'Algorithm Value': values['algorithm'],
                    'Reference Value': values['reference'],
                    'Relative Improvement (%)': values['relative_improvement'] * 100,
                    'Statistically Significant': result.statistical_tests['mann_whitney'].significant,
                    'Effect Size': result.statistical_tests['effect_size'],
                    'Better Than Reference': result.is_better
                })
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def _compute_convergence_speed(fitness_history: np.ndarray) -> float:
        """Compute convergence speed as the average improvement rate"""
        improvements = np.diff(fitness_history)
        return -np.mean(improvements[improvements < 0])
    
    @staticmethod
    def _compute_relative_improvement(
            alg_value: float,
            ref_value: float,
            better: str
        ) -> float:
        """Compute relative improvement over reference value"""
        if better == 'lower':
            return (ref_value - alg_value) / ref_value
        else:  # better == 'higher'
            return (alg_value - ref_value) / ref_value
    
    @staticmethod
    def _is_better_overall(metrics: Dict[str, Dict[str, float]]) -> bool:
        """Determine if algorithm is better overall based on metrics"""
        improvements = [m['relative_improvement'] for m in metrics.values()]
        return np.mean(improvements) > 0
    
    def _generate_comparison_description(
            self,
            metrics: Dict[str, Dict[str, float]],
            statistical_tests: Dict[str, Any],
            func_name: str
        ) -> str:
        """Generate a detailed comparison description"""
        better_metrics = [
            name for name, values in metrics.items()
            if values['relative_improvement'] > 0
        ]
        worse_metrics = [
            name for name, values in metrics.items()
            if values['relative_improvement'] < 0
        ]
        
        description = f"Function: {func_name}\n"
        
        if statistical_tests['mann_whitney'].significant:
            effect = self.statistical_analyzer._interpret_effect_size(
                statistical_tests['effect_size']
            )
            description += f"Statistically significant difference found ({effect})\n"
        else:
            description += "No statistically significant difference found\n"
        
        if better_metrics:
            description += f"Better in: {', '.join(better_metrics)}\n"
        if worse_metrics:
            description += f"Worse in: {', '.join(worse_metrics)}\n"
            
        return description.strip()
