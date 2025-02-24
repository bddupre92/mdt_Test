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
