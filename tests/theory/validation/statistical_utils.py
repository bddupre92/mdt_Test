"""
Statistical Analysis Utilities for Migraine Digital Twin Validation.

This module provides statistical analysis tools for validating theoretical
components and assessing test results.
"""

from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import auc
import pandas as pd

def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for a sample.
    
    Args:
        data: Sample data
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    return interval[0], interval[1]

def effect_size(
    group1: np.ndarray,
    group2: np.ndarray
) -> Dict[str, float]:
    """
    Calculate effect size metrics between two groups.
    
    Args:
        group1: First group's data
        group2: Second group's data
        
    Returns:
        Dictionary containing:
        - cohens_d: Cohen's d effect size
        - hedges_g: Hedges' g effect size
        - glass_delta: Glass's Δ effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Cohen's d
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    # Hedges' g (bias-corrected)
    hedges_g = cohens_d * (1 - (3 / (4 * (n1 + n2) - 9)))
    
    # Glass's Δ
    glass_delta = (np.mean(group1) - np.mean(group2)) / np.sqrt(var2)
    
    return {
        'cohens_d': cohens_d,
        'hedges_g': hedges_g,
        'glass_delta': glass_delta
    }

def power_analysis(
    effect_size: float,
    sample_size: int,
    alpha: float = 0.05,
    test_type: str = 'two-sided'
) -> float:
    """
    Calculate statistical power for a given effect size and sample size.
    
    Args:
        effect_size: Expected effect size (Cohen's d)
        sample_size: Sample size per group
        alpha: Significance level
        test_type: Type of test ('two-sided' or 'one-sided')
        
    Returns:
        Statistical power (1 - β)
    """
    if test_type not in ['two-sided', 'one-sided']:
        raise ValueError("test_type must be 'two-sided' or 'one-sided'")
    
    # Calculate non-centrality parameter
    ncp = effect_size * np.sqrt(sample_size / 2)
    
    # Calculate critical value
    if test_type == 'two-sided':
        crit = stats.norm.ppf(1 - alpha/2)
    else:
        crit = stats.norm.ppf(1 - alpha)
    
    # Calculate power
    if test_type == 'two-sided':
        power = (1 - stats.norm.cdf(crit - ncp) +
                stats.norm.cdf(-crit - ncp))
    else:
        power = 1 - stats.norm.cdf(crit - ncp)
    
    return power

def equivalence_test(
    group1: np.ndarray,
    group2: np.ndarray,
    margin: float,
    alpha: float = 0.05
) -> Dict[str, Union[bool, float]]:
    """
    Perform equivalence testing using two one-sided tests (TOST).
    
    Args:
        group1: First group's data
        group2: Second group's data
        margin: Equivalence margin
        alpha: Significance level
        
    Returns:
        Dictionary containing:
        - equivalent: Whether groups are equivalent
        - p_value: P-value for equivalence test
        - ci_lower: Lower confidence interval bound
        - ci_upper: Upper confidence interval bound
    """
    n1, n2 = len(group1), len(group2)
    mean_diff = np.mean(group1) - np.mean(group2)
    
    # Calculate standard error
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    se = np.sqrt(var1/n1 + var2/n2)
    
    # Calculate t-statistics for both one-sided tests
    t1 = (mean_diff - margin) / se
    t2 = (mean_diff + margin) / se
    
    # Calculate p-values
    df = n1 + n2 - 2
    p1 = 1 - stats.t.cdf(t1, df)
    p2 = stats.t.cdf(t2, df)
    
    # Overall p-value is the maximum of the two
    p_value = max(p1, p2)
    
    # Calculate confidence interval
    t_crit = stats.t.ppf(1 - alpha, df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    
    return {
        'equivalent': p_value < alpha,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def trend_analysis(
    time_series: np.ndarray,
    timestamps: Optional[np.ndarray] = None
) -> Dict[str, Union[float, bool]]:
    """
    Analyze trends in time series data.
    
    Args:
        time_series: Array of values over time
        timestamps: Optional array of timestamps
        
    Returns:
        Dictionary containing:
        - trend_coefficient: Linear trend coefficient
        - trend_p_value: P-value for trend significance
        - has_trend: Whether trend is significant (p < 0.05)
        - seasonal: Whether seasonal component is detected
        - stationarity: Whether series is stationary
    """
    if timestamps is None:
        timestamps = np.arange(len(time_series))
    
    # Linear trend analysis
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        timestamps, time_series
    )
    
    # Check for seasonality using autocorrelation
    acf = pd.Series(time_series).autocorr(lag=len(time_series)//4)
    seasonal = abs(acf) > 0.3
    
    # Check for stationarity using Augmented Dickey-Fuller test
    adf_stat, adf_p = stats.adfuller(time_series)[0:2]
    stationary = adf_p < 0.05
    
    return {
        'trend_coefficient': slope,
        'trend_p_value': p_value,
        'has_trend': p_value < 0.05,
        'seasonal': seasonal,
        'stationarity': stationary
    }

def cross_validation_stats(
    cv_scores: np.ndarray
) -> Dict[str, float]:
    """
    Calculate statistics for cross-validation results.
    
    Args:
        cv_scores: Array of cross-validation scores
        
    Returns:
        Dictionary containing:
        - mean_score: Mean CV score
        - std_score: Standard deviation of CV scores
        - ci_lower: Lower bound of 95% CI
        - ci_upper: Upper bound of 95% CI
        - cv_stability: Coefficient of variation
    """
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    ci_lower, ci_upper = confidence_interval(cv_scores)
    cv_stability = std_score / mean_score if mean_score != 0 else float('inf')
    
    return {
        'mean_score': mean_score,
        'std_score': std_score,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cv_stability': cv_stability
    }

def performance_comparison(
    baseline_scores: np.ndarray,
    new_scores: np.ndarray,
    min_improvement: float = 0.05
) -> Dict[str, Union[bool, float]]:
    """
    Compare performance between baseline and new implementation.
    
    Args:
        baseline_scores: Array of baseline performance scores
        new_scores: Array of new implementation scores
        min_improvement: Minimum improvement threshold
        
    Returns:
        Dictionary containing:
        - improved: Whether new implementation is significantly better
        - percent_improvement: Percentage improvement
        - p_value: P-value for improvement significance
        - effect_size: Cohen's d effect size
    """
    # Calculate improvement
    baseline_mean = np.mean(baseline_scores)
    new_mean = np.mean(new_scores)
    percent_improvement = (new_mean - baseline_mean) / baseline_mean
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(new_scores, baseline_scores)
    
    # Effect size
    effect = effect_size(new_scores, baseline_scores)
    
    return {
        'improved': (percent_improvement > min_improvement) and (p_value < 0.05),
        'percent_improvement': percent_improvement,
        'p_value': p_value,
        'effect_size': effect['cohens_d']
    } 