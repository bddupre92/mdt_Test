"""
Expert Evaluation Functions

This module provides specialized evaluation functions tailored for different expert domains
in the MoE framework. These functions incorporate domain-specific knowledge to better
assess model performance for each expert type.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def rmse_with_smoothness_penalty(y_true: np.ndarray, y_pred: np.ndarray, 
                                smoothness_weight: float = 0.2) -> float:
    """
    RMSE with a smoothness penalty for physiological data.
    
    Penalizes rapid oscillations in predictions which are physiologically unlikely.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        smoothness_weight: Weight for the smoothness penalty term
        
    Returns:
        Combined error score (lower is better)
    """
    # Base RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Smoothness penalty: penalize large differences between consecutive predictions
    if len(y_pred) > 1:
        diffs = np.diff(y_pred)
        smoothness_penalty = np.mean(np.square(diffs))
        return rmse + smoothness_weight * smoothness_penalty
    
    return rmse

def mae_with_lag_penalty(y_true: np.ndarray, y_pred: np.ndarray, 
                        lag_weight: float = 0.3,
                        max_lag: int = 3) -> float:
    """
    MAE with a penalty for prediction lag for environmental data.
    
    Environmental effects often have delayed impacts, this metric accounts for
    potential time-lagged correlations.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        lag_weight: Weight for the lag correlation term
        max_lag: Maximum lag to consider
        
    Returns:
        Combined error score (lower is better)
    """
    # Base MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # Lag penalty: check if predictions correlate better with time-shifted ground truth
    if len(y_true) > max_lag + 1:
        # Calculate correlation at different lags
        cors = []
        for lag in range(1, max_lag + 1):
            cor = np.corrcoef(y_true[lag:], y_pred[:-lag])[0, 1]
            if not np.isnan(cor):
                cors.append(cor)
        
        # If any lagged correlation is better than direct correlation, apply penalty
        if cors and max(cors) > np.corrcoef(y_true, y_pred)[0, 1]:
            lag_penalty = lag_weight * (max(cors) - np.corrcoef(y_true, y_pred)[0, 1])
            return mae + lag_penalty
    
    return mae

def weighted_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray, 
                     rmse_weight: float = 0.7,
                     sparse_penalty_threshold: float = 0.1) -> float:
    """
    Weighted combination of RMSE and MAE with a penalty for behavioral data.
    
    Behavioral data often has sparse but significant events, 
    this metric balances different error types.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        rmse_weight: Weight for RMSE vs MAE (higher means more RMSE influence)
        sparse_penalty_threshold: Threshold to consider an event as significant
        
    Returns:
        Combined error score (lower is better)
    """
    # Calculate base metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Weighted combination
    combined_error = rmse_weight * rmse + (1 - rmse_weight) * mae
    
    # Penalty for missing significant events
    significant_indices = np.where(y_true > np.mean(y_true) + sparse_penalty_threshold * np.std(y_true))[0]
    if len(significant_indices) > 0:
        significant_errors = np.abs(y_true[significant_indices] - y_pred[significant_indices])
        sparse_penalty = np.mean(significant_errors) - mae
        if sparse_penalty > 0:
            combined_error += 0.5 * sparse_penalty
    
    return combined_error

def treatment_response_score(y_true: np.ndarray, y_pred: np.ndarray,
                           treatment_indicators: Optional[np.ndarray] = None,
                           error_weight: float = 0.6,
                           response_weight: float = 0.4) -> float:
    """
    Specialized score for medication history data incorporating treatment response patterns.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        treatment_indicators: Binary array indicating treatment periods (if None, uses a heuristic)
        error_weight: Weight for the basic error term
        response_weight: Weight for the treatment response term
        
    Returns:
        Combined error score (lower is better)
    """
    # Basic error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # If no treatment indicators provided, try to infer patterns from the data
    if treatment_indicators is None and len(y_true) > 3:
        # Simple heuristic: look for sustained decrease patterns in ground truth
        diffs = np.diff(y_true)
        decreasing = diffs < 0
        # Identify sequences of multiple decreasing values
        treatment_indicators = np.zeros_like(y_true)
        for i in range(len(decreasing) - 1):
            if decreasing[i] and decreasing[i+1]:
                treatment_indicators[i:i+3] = 1
    
    # Treatment response term
    response_error = rmse
    if treatment_indicators is not None and np.any(treatment_indicators):
        # During treatment periods, prediction should reflect the actual response
        treatment_indices = np.where(treatment_indicators > 0)[0]
        if len(treatment_indices) > 0:
            treatment_error = np.sqrt(mean_squared_error(
                y_true[treatment_indices], y_pred[treatment_indices]))
            
            # Additional penalty if direction of change is wrong during treatment
            if len(treatment_indices) > 1:
                true_direction = np.sign(np.diff(y_true[treatment_indices]))
                pred_direction = np.sign(np.diff(y_pred[treatment_indices]))
                direction_matches = true_direction == pred_direction
                direction_penalty = 1.0 - np.mean(direction_matches)
                treatment_error *= (1.0 + direction_penalty)
            
            response_error = treatment_error
    
    # Combine scores
    return error_weight * rmse + response_weight * response_error

def create_evaluation_function(function_type: str, **kwargs) -> Callable:
    """
    Factory function to create an evaluation function of the specified type.
    
    Args:
        function_type: Type of evaluation function to create or expert type name
        **kwargs: Additional parameters for the evaluation function
        
    Returns:
        Evaluation function that takes y_true and y_pred arrays
    """
    # Direct mapping of function names to implementation
    function_map = {
        'rmse_with_smoothness_penalty': rmse_with_smoothness_penalty,
        'mae_with_lag_penalty': mae_with_lag_penalty,
        'weighted_rmse_mae': weighted_rmse_mae,
        'treatment_response_score': treatment_response_score
    }
    
    # Mapping from expert types to evaluation function names
    expert_type_map = {
        'physiological': 'rmse_with_smoothness_penalty',
        'environmental': 'mae_with_lag_penalty',
        'behavioral': 'weighted_rmse_mae',
        'medication_history': 'treatment_response_score'
    }
    
    # If function_type is an expert type, convert it to a function name
    if function_type in expert_type_map:
        function_type = expert_type_map[function_type]
    
    if function_type not in function_map:
        raise ValueError(f"Unknown evaluation function type: {function_type}. "
                        f"Available types: {list(function_map.keys())}" 
                        f" or expert types: {list(expert_type_map.keys())}")
    
    base_function = function_map[function_type]
    
    # Return a function that includes any additional parameters
    def configured_function(y_true, y_pred, **additional_kwargs):
        combined_kwargs = {**kwargs, **additional_kwargs}
        return base_function(y_true, y_pred, **combined_kwargs)
    
    return configured_function
