"""
statistical_analysis.py
---------------------
Framework for statistical analysis of optimization results.
Includes hypothesis testing, effect size calculation, and ranking methods.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Any
import pandas as pd
from dataclasses import dataclass

@dataclass
class StatisticalResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float = None
    significant: bool = None
    description: str = None

class StatisticalAnalyzer:
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha
    
    def mann_whitney_test(
            self,
            x: np.ndarray,
            y: np.ndarray,
            alternative: str = 'two-sided'
        ) -> StatisticalResult:
        """
        Perform Mann-Whitney U test for comparing two independent samples.
        
        Args:
            x: First sample
            y: Second sample
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            StatisticalResult with test details
        """
        statistic, p_value = stats.mannwhitneyu(x, y, alternative=alternative)
        effect_size = self._cliff_delta(x, y)
        
        return StatisticalResult(
            test_name='Mann-Whitney U test',
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            significant=p_value < self.alpha,
            description=self._interpret_effect_size(effect_size)
        )
    
    def friedman_test(self, results: np.ndarray) -> StatisticalResult:
        """
        Perform Friedman test for comparing multiple algorithms.
        
        Args:
            results: Array of shape (n_problems, n_algorithms) containing performance metrics
            
        Returns:
            StatisticalResult with test details
        """
        statistic, p_value = stats.friedmanchisquare(*[results[:, i] for i in range(results.shape[1])])
        
        return StatisticalResult(
            test_name='Friedman test',
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.alpha,
            description='Significant differences exist between algorithms' if p_value < self.alpha else 'No significant differences detected'
        )
    
    def wilcoxon_test(
            self,
            x: np.ndarray,
            y: np.ndarray,
            alternative: str = 'two-sided'
        ) -> StatisticalResult:
        """
        Perform Wilcoxon signed-rank test for paired samples.
        
        Args:
            x: First sample
            y: Second sample
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            StatisticalResult with test details
        """
        statistic, p_value = stats.wilcoxon(x, y, alternative=alternative)
        effect_size = self._rank_biserial_correlation(x, y)
        
        return StatisticalResult(
            test_name='Wilcoxon signed-rank test',
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            significant=p_value < self.alpha,
            description=self._interpret_effect_size(effect_size)
        )
    
    def compare_algorithms(
            self,
            results: Dict[str, Dict[str, List[float]]],
            paired: bool = True
        ) -> pd.DataFrame:
        """
        Perform comprehensive comparison between algorithms.
        
        Args:
            results: Nested dictionary {function_name: {algorithm_name: [runs]}}
            paired: Whether to use paired tests (Wilcoxon) or unpaired (Mann-Whitney)
            
        Returns:
            DataFrame with pairwise comparison results
        """
        comparisons = []
        
        for func_name, func_results in results.items():
            alg_names = list(func_results.keys())
            
            for i in range(len(alg_names)):
                for j in range(i + 1, len(alg_names)):
                    alg1, alg2 = alg_names[i], alg_names[j]
                    x = np.array(func_results[alg1])
                    y = np.array(func_results[alg2])
                    
                    if paired:
                        result = self.wilcoxon_test(x, y)
                    else:
                        result = self.mann_whitney_test(x, y)
                    
                    comparisons.append({
                        'Function': func_name,
                        'Algorithm 1': alg1,
                        'Algorithm 2': alg2,
                        'Test': result.test_name,
                        'p-value': result.p_value,
                        'Effect Size': result.effect_size,
                        'Significant': result.significant,
                        'Description': result.description
                    })
        
        return pd.DataFrame(comparisons)
    
    @staticmethod
    def _cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Cliff's Delta effect size"""
        nx, ny = len(x), len(y)
        dominance = 0
        
        for i in x:
            for j in y:
                if i > j:
                    dominance += 1
                elif i < j:
                    dominance -= 1
                    
        return dominance / (nx * ny)
    
    @staticmethod
    def _rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate rank biserial correlation effect size"""
        n = len(x)
        pos_ranks_sum = sum(1 for d in y - x if d > 0)
        neg_ranks_sum = sum(1 for d in y - x if d < 0)
        
        return (pos_ranks_sum - neg_ranks_sum) / (pos_ranks_sum + neg_ranks_sum)
    
    @staticmethod
    def _interpret_effect_size(effect_size: float) -> str:
        """Interpret effect size magnitude"""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.147:
            magnitude = "negligible"
        elif abs_effect < 0.33:
            magnitude = "small"
        elif abs_effect < 0.474:
            magnitude = "medium"
        else:
            magnitude = "large"
            
        direction = "positive" if effect_size > 0 else "negative"
        return f"{magnitude} {direction} effect"

def run_statistical_tests(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive statistical analysis on optimization results.
    
    Args:
        results: Dictionary containing optimization results
        
    Returns:
        Dictionary with statistical analysis results
    """
    analyzer = StatisticalAnalyzer()
    analysis = {}
    
    # Analyze each mode
    for mode in results:
        analysis[mode] = {}
        
        # Analyze each test suite
        for suite_name, suite_results in results[mode].items():
            suite_analysis = {}
            
            # Analyze each function
            for func_name, trials in suite_results.items():
                # Extract final scores
                scores = [trial['value'] for trial in trials]
                
                # Basic statistics
                basic_stats = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'median': float(np.median(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                }
                
                # Convergence analysis
                convergence_stats = analyze_convergence(trials)
                
                # Selection stability
                stability_stats = analyze_selection_stability(trials)
                
                # Combine all statistics
                suite_analysis[func_name] = {
                    'basic_stats': basic_stats,
                    'convergence': convergence_stats,
                    'stability': stability_stats
                }
            
            analysis[mode][suite_name] = suite_analysis
    
    # Compare modes
    analysis['mode_comparison'] = compare_modes(results)
    
    return analysis

def analyze_convergence(trials: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze convergence behavior across trials"""
    # Extract convergence histories
    histories = [trial['history']['scores'] for trial in trials]
    
    # Calculate convergence metrics
    convergence_rates = []
    for history in histories:
        # Use log-log slope as convergence rate
        x = np.log(np.arange(1, len(history) + 1))
        y = np.log(history)
        slope, _, r_value, _, _ = stats.linregress(x, y)
        convergence_rates.append(slope)
    
    return {
        'mean_conv_rate': float(np.mean(convergence_rates)),
        'std_conv_rate': float(np.std(convergence_rates)),
        'r_squared': float(r_value**2)
    }

def analyze_selection_stability(trials: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze stability of optimizer selection"""
    # Extract optimizer selections
    selections = [trial['history']['optimizers'] for trial in trials]
    
    # Calculate transition matrices
    transition_counts = {}
    for trial_selections in selections:
        for i in range(len(trial_selections)-1):
            key = (trial_selections[i], trial_selections[i+1])
            transition_counts[key] = transition_counts.get(key, 0) + 1
    
    # Calculate stability metrics
    unique_optimizers = set()
    for trial_selections in selections:
        unique_optimizers.update(trial_selections)
    
    return {
        'n_optimizers_used': len(unique_optimizers),
        'mean_transitions': float(np.mean([len(set(s)) for s in selections])),
        'selection_entropy': float(stats.entropy([
            sum(1 for s in selections for opt in s if opt == optimizer)
            for optimizer in unique_optimizers
        ]))
    }

def compare_modes(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare different meta-learning modes"""
    mode_comparison = {}
    
    # Get all modes
    modes = list(results.keys())
    
    # Compare each pair of modes
    for i, mode1 in enumerate(modes):
        for mode2 in modes[i+1:]:
            comparison = {}
            
            # Compare on each function
            for suite in results[mode1]:
                for func in results[mode1][suite]:
                    scores1 = [
                        trial['value'] 
                        for trial in results[mode1][suite][func]
                    ]
                    scores2 = [
                        trial['value'] 
                        for trial in results[mode2][suite][func]
                    ]
                    
                    # Perform statistical test
                    stat, p_value = stats.mannwhitneyu(scores1, scores2)
                    effect_size = np.abs(
                        np.mean(scores1) - np.mean(scores2)
                    ) / np.std(scores1 + scores2)
                    
                    comparison[f"{suite}/{func}"] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'effect_size': float(effect_size)
                    }
            
            mode_comparison[f"{mode1}_vs_{mode2}"] = comparison
    
    return mode_comparison
