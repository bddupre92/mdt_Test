"""
Prediction metrics for evaluating model performance.

This module provides metrics for evaluating the accuracy, reliability,
and calibration of prediction models in the migraine prediction system.
"""

from typing import Dict, List, Tuple, Any, Union, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, precision_recall_curve, average_precision_score
)
import scipy.stats as stats

def binary_prediction_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate metrics for binary prediction models.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities or scores
        threshold: Classification threshold for predicted probabilities
        
    Returns:
        Dictionary of classification metrics
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # For probability outputs, convert to binary predictions
    if np.any((y_pred > 0) & (y_pred < 1)):
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred
    
    # Calculate metrics
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred_binary)
    metrics["precision"] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # AUC-ROC (needs probability scores)
    if np.any((y_pred > 0) & (y_pred < 1)):
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
            metrics["average_precision"] = average_precision_score(y_true, y_pred)
        except ValueError:
            metrics["auc_roc"] = 0.5  # Default for failed calculation
            metrics["average_precision"] = y_true.mean()  # Base rate
    else:
        metrics["auc_roc"] = 0.5  # Default for non-probabilistic outputs
        metrics["average_precision"] = y_true.mean()  # Base rate
    
    # Calculate positive and negative predictive values
    true_positives = np.sum((y_true == 1) & (y_pred_binary == 1))
    false_positives = np.sum((y_true == 0) & (y_pred_binary == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred_binary == 0))
    false_negatives = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    # Positive predictive value (precision)
    metrics["ppv"] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # Negative predictive value
    metrics["npv"] = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
    
    return metrics

def regression_metrics(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray]
) -> Dict[str, float]:
    """Calculate metrics for regression prediction models.
    
    Args:
        y_true: True continuous values
        y_pred: Predicted continuous values
        
    Returns:
        Dictionary of regression metrics
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Calculate metrics
    metrics = {}
    
    # Error metrics
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["r2"] = r2_score(y_true, y_pred)
    
    # Calculate normalized metrics
    y_range = np.max(y_true) - np.min(y_true)
    if y_range > 0:
        metrics["normalized_mae"] = metrics["mae"] / y_range
        metrics["normalized_rmse"] = metrics["rmse"] / y_range
    else:
        metrics["normalized_mae"] = 0.0
        metrics["normalized_rmse"] = 0.0
    
    # Calculate correlation
    if len(y_true) > 1:
        metrics["correlation"] = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        metrics["correlation"] = 0.0
    
    return metrics

def confidence_calibration(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    confidences: Union[List[float], np.ndarray],
    n_bins: int = 10
) -> Dict[str, Any]:
    """Evaluate the calibration of confidence estimates.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels or probabilities
        confidences: Confidence values for predictions (0-1)
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary of calibration metrics
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    if not isinstance(confidences, np.ndarray):
        confidences = np.array(confidences)
    
    # For probability outputs, convert to binary predictions
    if np.any((y_pred > 0) & (y_pred < 1)):
        y_pred_binary = (y_pred >= 0.5).astype(int)
    else:
        y_pred_binary = y_pred
    
    # Bin the confidence values
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure valid bin indices
    
    # Calculate calibration metrics per bin
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_idx in range(n_bins):
        mask = (bin_indices == bin_idx)
        if np.sum(mask) > 0:
            bin_accuracy = accuracy_score(y_true[mask], y_pred_binary[mask])
            bin_confidence = np.mean(confidences[mask])
            bin_count = np.sum(mask)
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
    
    # Calculate overall metrics
    calibration_error = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences))) if bin_accuracies else 1.0
    
    # Calculate coverage at different confidence thresholds
    thresholds = [0.9, 0.95, 0.99]
    coverage = {}
    for threshold in thresholds:
        high_conf_mask = (confidences >= threshold)
        if np.sum(high_conf_mask) > 0:
            coverage[f"accuracy_at_{threshold}"] = accuracy_score(y_true[high_conf_mask], y_pred_binary[high_conf_mask])
            coverage[f"coverage_at_{threshold}"] = np.mean(high_conf_mask)
        else:
            coverage[f"accuracy_at_{threshold}"] = 0.0
            coverage[f"coverage_at_{threshold}"] = 0.0
    
    return {
        "calibration_error": calibration_error,
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "coverage": coverage
    }

def forecast_reliability(
    forecast_horizons: List[int],
    accuracies: List[float],
    confidences: List[float],
    decay_factor: float = 0.1
) -> Dict[str, Any]:
    """Evaluate the reliability of forecasts across different time horizons.
    
    Args:
        forecast_horizons: List of forecast horizons (hours)
        accuracies: Corresponding accuracy values for each horizon
        confidences: Corresponding confidence values for each horizon
        decay_factor: Expected decay rate in accuracy per unit time
        
    Returns:
        Dictionary of forecast reliability metrics
    """
    if not forecast_horizons or len(forecast_horizons) != len(accuracies) or len(forecast_horizons) != len(confidences):
        return {
            "horizon_reliability": {},
            "overall_reliability": 0.0,
            "max_reliable_horizon": 0,
            "reliability_curve": {"horizons": [], "reliabilities": []}
        }
    
    # Convert to numpy arrays
    horizons = np.array(forecast_horizons)
    acc = np.array(accuracies)
    conf = np.array(confidences)
    
    # Calculate the expected accuracy based on decay model
    expected_acc = np.exp(-decay_factor * horizons)
    
    # Calculate reliability metrics for each horizon
    horizon_reliability = {}
    for i, horizon in enumerate(horizons):
        # Reliability is the ratio of actual accuracy to expected accuracy
        reliability = acc[i] / expected_acc[i] if expected_acc[i] > 0 else 0.0
        
        # Calculate calibration error at this horizon
        calibration_error = abs(acc[i] - conf[i])
        
        horizon_reliability[str(horizon)] = {
            "reliability": reliability,
            "accuracy": acc[i],
            "confidence": conf[i],
            "expected_accuracy": expected_acc[i],
            "calibration_error": calibration_error
        }
    
    # Calculate overall reliability as weighted average
    weights = 1.0 / (1.0 + horizons)  # More weight to shorter horizons
    overall_reliability = np.sum(acc * weights) / np.sum(weights)
    
    # Find the maximum reliable horizon (where accuracy is at least 0.7 of expected)
    reliable_mask = acc >= (0.7 * expected_acc)
    max_reliable_horizon = np.max(horizons[reliable_mask]) if np.any(reliable_mask) else 0
    
    # Generate reliability curve
    sorted_indices = np.argsort(horizons)
    sorted_horizons = horizons[sorted_indices]
    reliabilities = acc[sorted_indices] / expected_acc[sorted_indices]
    
    return {
        "horizon_reliability": horizon_reliability,
        "overall_reliability": float(overall_reliability),
        "max_reliable_horizon": int(max_reliable_horizon),
        "reliability_curve": {
            "horizons": sorted_horizons.tolist(),
            "reliabilities": reliabilities.tolist()
        }
    }

def drift_detection_metrics(
    time_periods: List[str],
    distributions: List[np.ndarray],
    reference_distribution: np.ndarray,
    drift_threshold: float = 0.05
) -> Dict[str, Any]:
    """Calculate metrics for detecting distribution drift over time.
    
    Args:
        time_periods: List of time period identifiers
        distributions: List of data distributions for each time period
        reference_distribution: Reference distribution for comparison
        drift_threshold: P-value threshold for statistical significance
        
    Returns:
        Dictionary of drift detection metrics
    """
    if not time_periods or len(time_periods) != len(distributions):
        return {
            "drift_detected": False,
            "drift_points": [],
            "drift_magnitudes": {},
            "stability_score": 1.0
        }
    
    # Calculate drift for each time period
    drift_points = []
    drift_magnitudes = {}
    p_values = []
    
    for i, (period, dist) in enumerate(zip(time_periods, distributions)):
        # Use Kolmogorov-Smirnov test to detect distribution shifts
        ks_stat, p_value = stats.ks_2samp(dist, reference_distribution)
        p_values.append(p_value)
        
        # Check if drift is detected (p-value below threshold)
        if p_value < drift_threshold:
            drift_points.append(period)
            drift_magnitudes[period] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "significant": True
            }
        else:
            drift_magnitudes[period] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "significant": False
            }
    
    # Calculate stability score (1.0 = perfect stability, 0.0 = complete drift)
    stability_score = np.mean([1.0 - (1.0 if p < drift_threshold else 0.0) for p in p_values])
    
    return {
        "drift_detected": len(drift_points) > 0,
        "drift_points": drift_points,
        "drift_magnitudes": drift_magnitudes,
        "stability_score": float(stability_score)
    } 