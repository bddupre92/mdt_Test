"""
Metrics calculation utilities for baseline comparisons.

This module provides functions to calculate standard performance metrics
for regression tasks, allowing consistent evaluation across different approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_baseline_metrics(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series],
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute standard performance metrics for regression predictions.
    
    Args:
        y_true: Ground truth target values
        y_pred: Predicted target values
        prefix: Optional prefix for metric names in the output dictionary
    
    Returns:
        Dictionary of metric names to metric values
    """
    # Convert inputs to numpy arrays if they're pandas objects
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Calculate metrics
    metrics = {}
    
    # Root mean squared error
    metrics[f"{prefix}rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean absolute error
    metrics[f"{prefix}mae"] = mean_absolute_error(y_true, y_pred)
    
    # R-squared
    metrics[f"{prefix}r2"] = r2_score(y_true, y_pred)
    
    # Mean absolute percentage error (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics[f"{prefix}mape"] = mape if not np.isnan(mape) else np.inf
    
    return metrics


def aggregate_metrics(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate multiple metrics dictionaries.
    
    Args:
        metrics_list: List of metric dictionaries to aggregate
    
    Returns:
        Dictionary of aggregated metrics (mean values)
    """
    if not metrics_list:
        return {}
    
    # Initialize aggregated metrics with keys from first dictionary
    aggregated = {key: [] for key in metrics_list[0].keys()}
    
    # Collect values for each metric
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key in aggregated:
                aggregated[key].append(value)
    
    # Calculate mean for each metric
    result = {key: np.mean(values) for key, values in aggregated.items()}
    
    # Add standard deviation for each metric
    for key, values in aggregated.items():
        if len(values) > 1:
            result[f"{key}_std"] = np.std(values)
    
    return result


def compare_algorithms(
    y_true: Union[np.ndarray, pd.Series],
    predictions: Dict[str, np.ndarray],
    algorithms: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple algorithms based on their predictions.
    
    Args:
        y_true: Ground truth target values
        predictions: Dictionary mapping algorithm names to their predictions
        algorithms: Optional list of algorithm names to include in comparison
                   (defaults to all keys in predictions)
    
    Returns:
        Dictionary mapping algorithm names to their performance metrics
    """
    # Use all algorithms if none specified
    if algorithms is None:
        algorithms = list(predictions.keys())
    
    # Calculate metrics for each algorithm
    results = {}
    for algorithm in algorithms:
        if algorithm in predictions:
            results[algorithm] = compute_baseline_metrics(
                y_true=y_true,
                y_pred=predictions[algorithm]
            )
    
    return results
