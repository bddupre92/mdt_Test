"""
Theoretical Metrics for MoE Framework

This module implements advanced theoretical metrics for analyzing and
understanding the MoE framework's mathematical properties.

Key components:
- Convergence rate calculations for evolutionary algorithms
- Algorithm complexity scaling analysis
- Information-theoretic measures (transfer entropy, mutual information)
- Causal relationship extraction
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats
from scipy.optimize import curve_fit  # Add correct import for curve_fit
from sklearn.metrics import mutual_info_score
from statsmodels.tsa.stattools import grangercausalitytests
import time
from typing import Dict, List, Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)

def calculate_convergence_rate(optimization_trajectory: np.ndarray) -> Dict[str, float]:
    """
    Calculate convergence rate from optimization trajectory.
    
    Parameters:
    -----------
    optimization_trajectory : np.ndarray
        Array of fitness values across iterations
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with convergence metrics:
        - 'asymptotic_rate': Estimated asymptotic convergence rate
        - 'linear_fit_slope': Slope of log-error vs iterations
        - 'convergence_order': Estimated order of convergence (linear, quadratic)
    """
    if len(optimization_trajectory) < 3:
        logger.warning("Trajectory too short for convergence analysis")
        return {
            'asymptotic_rate': np.nan,
            'linear_fit_slope': np.nan,
            'convergence_order': np.nan
        }
    
    # Get the error as distance from optimum (using final value as proxy for optimum)
    optimum = optimization_trajectory[-1]
    errors = np.abs(optimization_trajectory - optimum)
    errors = np.maximum(errors, 1e-10)  # Avoid log(0)
    
    # Calculate log errors for rate analysis
    log_errors = np.log(errors)
    iterations = np.arange(len(optimization_trajectory))
    
    # Linear fit for exponential convergence rate
    valid_indices = np.isfinite(log_errors)
    if np.sum(valid_indices) < 2:
        slope = np.nan
    else:
        slope, _, _, _, _ = stats.linregress(
            iterations[valid_indices], 
            log_errors[valid_indices]
        )
    
    # Estimate order of convergence
    if len(errors) >= 3:
        # For sequential steps, convergence order is:
        # p ≈ log(|e_{k+2} - e_{k+1}|/|e_{k+1} - e_{k}|) / log(|e_{k+1} - e_{k}|/|e_{k} - e_{k-1}|)
        ratios = []
        for i in range(len(errors) - 2):
            if errors[i] > 1e-10 and errors[i+1] > 1e-10:
                ratio1 = np.abs((errors[i+2] - errors[i+1]) / (errors[i+1] - errors[i]))
                ratio2 = np.abs((errors[i+1] - errors[i]) / (errors[i] - errors[i-1]) if i > 0 else 1.0)
                if ratio2 > 1e-10 and ratio2 != 1.0:
                    log_ratio2 = np.log(ratio2)
                    # Prevent division by zero or very small values
                    if np.abs(log_ratio2) > 1e-10:
                        ratios.append(np.log(ratio1) / log_ratio2)
        
        order = np.median(ratios) if ratios else np.nan
    else:
        order = np.nan
    
    return {
        'asymptotic_rate': np.exp(slope),
        'linear_fit_slope': slope,
        'convergence_order': order
    }

def analyze_complexity_scaling(dimensionality_range: List[int], 
                              runtime_data: List[float]) -> Dict[str, Any]:
    """
    Analyze how algorithm complexity scales with dimensionality.
    
    Parameters:
    -----------
    dimensionality_range : List[int]
        List of dimensionality values tested
    runtime_data : List[float]
        Corresponding runtimes
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with complexity analysis:
        - 'complexity_class': Estimated complexity class (O(n), O(n²), etc.)
        - 'fitted_coefficients': Coefficients of polynomial fit
        - 'r_squared': R² of the fit
        - 'fit_curve': Function to predict runtime for given dimension
    """
    if len(dimensionality_range) < 2:
        logger.warning("Not enough data points for complexity analysis")
        return {
            'complexity_class': 'unknown',
            'fitted_coefficients': [],
            'r_squared': np.nan,
            'fit_curve': lambda x: np.nan
        }
    
    # Convert to numpy arrays
    dims = np.array(dimensionality_range)
    runtimes = np.array(runtime_data)
    
    # Try different complexity classes (linear, quadratic, etc.)
    complexity_models = {
        'O(1)': lambda x, a: a * np.ones_like(x),
        'O(log n)': lambda x, a, b: a * np.log(x) + b,
        'O(n)': lambda x, a, b: a * x + b,
        'O(n log n)': lambda x, a, b: a * x * np.log(x) + b,
        'O(n²)': lambda x, a, b: a * x**2 + b,
        'O(n³)': lambda x, a, b: a * x**3 + b,
        'O(2^n)': lambda x, a, b: a * 2**x + b
    }
    
    best_r2 = -np.inf
    best_model = 'unknown'
    best_params = []
    best_fit_curve = lambda x: np.nan
    
    # Fit each model and find the best one
    for name, model in complexity_models.items():
        try:
            if name == 'O(1)':
                popt, _ = curve_fit(model, dims, runtimes)
                predicted = model(dims, *popt)
            else:
                popt, _ = curve_fit(model, dims, runtimes)
                predicted = model(dims, *popt)
            
            r2 = 1 - np.sum((runtimes - predicted)**2) / np.sum((runtimes - np.mean(runtimes))**2)
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = name
                best_params = popt
                best_fit_curve = lambda x, m=model, p=popt: m(x, *p)
                
        except Exception as e:
            logger.warning(f"Failed to fit {name} model: {e}")
    
    return {
        'complexity_class': best_model,
        'fitted_coefficients': best_params,
        'r_squared': best_r2,
        'fit_curve': best_fit_curve
    }

def calculate_transfer_entropy(source_series: np.ndarray, 
                              target_series: np.ndarray, 
                              lags: int = 1, 
                              bins: Optional[int] = None) -> float:
    """
    Calculate transfer entropy from source to target time series.
    Transfer entropy measures directed information flow.
    
    Parameters:
    -----------
    source_series : np.ndarray
        Source time series
    target_series : np.ndarray
        Target time series
    lags : int
        Number of lags to consider
    bins : Optional[int]
        Number of bins for discretization, if None will use Sturges' rule
        
    Returns:
    --------
    float
        Transfer entropy value (in bits)
    """
    if len(source_series) != len(target_series):
        raise ValueError("Source and target series must have same length")
    
    if len(source_series) <= lags + 1:
        logger.warning("Time series too short for transfer entropy calculation")
        return 0.0
    
    # Discretize if needed
    if bins is None:
        bins = int(np.ceil(np.log2(len(source_series)) + 1))  # Sturges' rule
    
    # Discretize the data
    source_discrete = pd.qcut(source_series, bins, labels=False, duplicates='drop')
    target_discrete = pd.qcut(target_series, bins, labels=False, duplicates='drop')
    
    if len(np.unique(source_discrete)) < 2 or len(np.unique(target_discrete)) < 2:
        logger.warning("Not enough unique values after discretization")
        return 0.0
    
    # Create lagged versions
    target_future = target_discrete[lags:]
    target_past = target_discrete[:-lags]
    source_past = source_discrete[:-lags]
    
    # Calculate entropies
    h_target_past = stats.entropy(np.histogram(target_past, bins=np.unique(target_past).size)[0])
    h_source_target_past = stats.entropy(*np.histogramdd([source_past, target_past], 
                                                      bins=[np.unique(source_past).size, 
                                                            np.unique(target_past).size])[0].flatten())
    
    # Joint entropy of target future and target past
    joint_counts_tp_tf = np.zeros((np.unique(target_past).size, np.unique(target_future).size))
    for i, j in zip(target_past, target_future):
        joint_counts_tp_tf[i, j] += 1
    joint_prob_tp_tf = joint_counts_tp_tf / np.sum(joint_counts_tp_tf)
    h_target_past_future = -np.sum(joint_prob_tp_tf * np.log2(joint_prob_tp_tf + 1e-10))
    
    # Joint entropy of source past, target past and target future
    joint_counts_sp_tp_tf = np.zeros((np.unique(source_past).size, 
                                    np.unique(target_past).size, 
                                    np.unique(target_future).size))
    
    for i, j, k in zip(source_past, target_past, target_future):
        joint_counts_sp_tp_tf[i, j, k] += 1
    joint_prob_sp_tp_tf = joint_counts_sp_tp_tf / np.sum(joint_counts_sp_tp_tf)
    h_source_target_past_future = -np.sum(joint_prob_sp_tp_tf * np.log2(joint_prob_sp_tp_tf + 1e-10))
    
    # Transfer entropy
    te = h_target_past_future - h_source_target_past_future + h_source_target_past - h_target_past
    
    return max(0, te)  # Transfer entropy should be non-negative

def measure_mutual_information(X: np.ndarray, 
                              Y: np.ndarray, 
                              normalized: bool = True) -> float:
    """
    Calculate mutual information between two variables.
    
    Parameters:
    -----------
    X : np.ndarray
        First variable
    Y : np.ndarray
        Second variable
    normalized : bool
        If True, normalize to [0,1] range
        
    Returns:
    --------
    float
        Mutual information value
    """
    if len(X) != len(Y):
        raise ValueError("Arrays must have same length")
    
    # Discretize continuous variables if needed
    if not np.issubdtype(X.dtype, np.integer):
        bins_x = min(int(np.sqrt(len(X))), 30)  # Rule of thumb
        X = pd.qcut(X, bins_x, labels=False, duplicates='drop')
    
    if not np.issubdtype(Y.dtype, np.integer):
        bins_y = min(int(np.sqrt(len(Y))), 30)
        Y = pd.qcut(Y, bins_y, labels=False, duplicates='drop')
    
    # Calculate mutual information
    mi = mutual_info_score(X, Y)
    
    # Normalize if requested
    if normalized:
        h_x = stats.entropy(np.bincount(X))
        h_y = stats.entropy(np.bincount(Y))
        
        if min(h_x, h_y) > 0:
            mi /= np.sqrt(h_x * h_y)
        else:
            mi = 0  # No uncertainty in one of the variables
    
    return mi

def extract_causal_relationships(data: pd.DataFrame, 
                                target_col: str, 
                                method: str = 'granger', 
                                max_lag: int = 5, 
                                significance: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """
    Extract causal relationships between features and target variables.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with time series data
    target_col : str
        Column name of the target variable
    method : str
        Causality test method ('granger' or 'correlation')
    max_lag : int
        Maximum lag to test for Granger causality
    significance : float
        Significance level for causal relationship detection
        
    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary of causal relationships with metrics
    """
    results = {}
    target = data[target_col].values
    
    if method == 'granger':
        # Test each feature for Granger causality with the target
        for col in data.columns:
            if col == target_col:
                continue
                
            try:
                # Prepare data - both series need to be stationary
                # For simplicity, we'll use differencing
                X = data[[col, target_col]].diff().dropna()
                
                # Run Granger causality test
                gc_res = grangercausalitytests(X, maxlag=max_lag, verbose=False)
                
                # Extract p-values for each lag
                p_values = {lag: test[0]['ssr_chi2test'][1] for lag, test in gc_res.items()}
                min_p_value = min(p_values.values())
                significant_lags = [lag for lag, p in p_values.items() if p < significance]
                
                results[col] = {
                    'causal': min_p_value < significance,
                    'p_value': min_p_value,
                    'significant_lags': significant_lags,
                    'strength': 1 - min_p_value if min_p_value < 1 else 0
                }
            except Exception as e:
                logger.warning(f"Error in Granger causality test for {col}: {e}")
                results[col] = {
                    'causal': False,
                    'p_value': 1.0,
                    'significant_lags': [],
                    'strength': 0,
                    'error': str(e)
                }
    
    elif method == 'correlation':
        # Use lagged correlation as a simple alternative
        for col in data.columns:
            if col == target_col:
                continue
                
            try:
                feature = data[col].values
                
                # Calculate lagged correlations
                correlations = []
                for lag in range(1, max_lag + 1):
                    corr = np.corrcoef(feature[:-lag], target[lag:])[0, 1]
                    correlations.append((lag, corr))
                
                # Find maximum correlation and its lag
                max_lag, max_corr = max(correlations, key=lambda x: abs(x[1]))
                
                results[col] = {
                    'causal': abs(max_corr) > 0.5,  # Arbitrary threshold
                    'correlation': max_corr,
                    'optimal_lag': max_lag,
                    'strength': abs(max_corr)
                }
            except Exception as e:
                logger.warning(f"Error in correlation analysis for {col}: {e}")
                results[col] = {
                    'causal': False,
                    'correlation': 0,
                    'optimal_lag': 0,
                    'strength': 0,
                    'error': str(e)
                }
    else:
        raise ValueError(f"Unknown causality method: {method}")
    
    return results

def analyze_temporal_patterns(data: pd.DataFrame, 
                             target_col: str,
                             threshold: float = 0.7) -> Dict[str, Any]:
    """
    Extract temporal patterns and potential triggers from time series data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with time series data
    target_col : str
        Column name of the target variable
    threshold : float
        Correlation threshold for pattern detection
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with temporal pattern analysis
    """
    results = {
        'periodic_features': [],
        'leading_indicators': [],
        'trigger_candidates': [],
        'recurring_patterns': {}
    }
    
    target = data[target_col].values
    
    # Detect periodicity in each feature
    for col in data.columns:
        if col == target_col:
            continue
            
        try:
            feature = data[col].values
            
            # Autocorrelation test for periodicity
            acf = np.correlate(feature, feature, mode='full')
            acf = acf[len(acf)//2:]  # Take only positive lags
            acf /= acf[0]  # Normalize
            
            # Find peaks in autocorrelation
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(acf, height=0.5)
            
            if len(peaks) > 0:
                dominant_period = peaks[0]
                results['periodic_features'].append({
                    'feature': col,
                    'period': dominant_period,
                    'acf_strength': acf[dominant_period]
                })
            
            # Calculate cross-correlation with target
            ccf = np.correlate(target, feature, mode='full')
            ccf = ccf[len(ccf)//2:]  # Take only positive lags
            ccf /= np.sqrt(np.sum(target**2) * np.sum(feature**2))  # Normalize
            
            # Find peaks in cross-correlation
            peaks, _ = find_peaks(ccf, height=threshold)
            
            if len(peaks) > 0:
                lead_time = peaks[0]
                if lead_time > 0:  # Feature leads target
                    results['leading_indicators'].append({
                        'feature': col,
                        'lead_time': lead_time,
                        'correlation': ccf[lead_time]
                    })
                    
                    # Strong leading indicators are potential triggers
                    if ccf[lead_time] > threshold:
                        results['trigger_candidates'].append({
                            'feature': col,
                            'lead_time': lead_time,
                            'strength': ccf[lead_time]
                        })
        except Exception as e:
            logger.warning(f"Error in temporal pattern analysis for {col}: {e}")
    
    # Detect recurring patterns in target variable
    try:
        # Use change points to detect patterns
        target_diff = np.diff(target)
        changes = np.where(np.abs(target_diff) > np.std(target_diff))[0]
        
        if len(changes) > 2:
            intervals = np.diff(changes)
            unique_intervals, counts = np.unique(intervals, return_counts=True)
            
            # Identify recurring intervals
            for interval, count in zip(unique_intervals, counts):
                if count > 1:  # Happens more than once
                    results['recurring_patterns'][int(interval)] = int(count)
    except Exception as e:
        logger.warning(f"Error in recurring pattern detection: {e}")
    
    return results

def benchmark_algorithm_complexity(algorithm, 
                                  input_sizes: List[int], 
                                  data_generator: callable,
                                  repetitions: int = 3) -> Dict[str, Any]:
    """
    Benchmark the algorithmic complexity of a function empirically.
    
    Parameters:
    -----------
    algorithm : callable
        The algorithm to benchmark
    input_sizes : List[int]
        List of input sizes to test
    data_generator : callable
        Function that generates input data of given size
    repetitions : int
        Number of repetitions for each size
        
    Returns:
    --------
    Dict[str, Any]
        Complexity analysis results
    """
    sizes = []
    times = []
    
    for size in input_sizes:
        runtime = []
        for _ in range(repetitions):
            data = data_generator(size)
            start_time = time.time()
            algorithm(data)
            end_time = time.time()
            runtime.append(end_time - start_time)
        
        sizes.append(size)
        times.append(np.median(runtime))
    
    # Analyze the complexity
    complexity_results = analyze_complexity_scaling(sizes, times)
    
    return {
        'input_sizes': sizes,
        'runtimes': times,
        'complexity_analysis': complexity_results
    }
