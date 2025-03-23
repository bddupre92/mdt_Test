"""
Clinical metrics for migraine prediction system validation.

This module provides metrics for evaluating the clinical relevance and utility
of the migraine prediction system from a patient and healthcare perspective.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from scipy import stats

def prediction_lead_time(
    predicted_times: np.ndarray,
    actual_times: np.ndarray,
    min_confidence: float = 0.8
) -> Dict[str, float]:
    """
    Calculate metrics related to how early migraines are predicted.
    
    Args:
        predicted_times: Array of predicted migraine onset times (timestamps)
        actual_times: Array of actual migraine onset times (timestamps)
        min_confidence: Minimum confidence threshold for predictions
        
    Returns:
        Dictionary containing:
        - mean_lead_time: Average time between prediction and onset
        - min_lead_time: Minimum lead time observed
        - max_lead_time: Maximum lead time observed
        - lead_time_std: Standard deviation of lead times
        - early_warning_rate: Proportion of migraines predicted with sufficient lead time
    """
    lead_times = actual_times - predicted_times
    
    return {
        'mean_lead_time': np.mean(lead_times),
        'min_lead_time': np.min(lead_times),
        'max_lead_time': np.max(lead_times),
        'lead_time_std': np.std(lead_times),
        'early_warning_rate': np.mean(lead_times >= 3600)  # 1 hour minimum
    }

def trigger_identification_accuracy(
    identified_triggers: List[str],
    known_triggers: List[str],
    trigger_intensities: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Calculate accuracy metrics for trigger identification.
    
    Args:
        identified_triggers: List of triggers identified by the system
        known_triggers: List of clinically confirmed triggers
        trigger_intensities: Optional dict of trigger intensity scores
        
    Returns:
        Dictionary containing:
        - precision: Proportion of identified triggers that are correct
        - recall: Proportion of known triggers that were identified
        - f1_score: Harmonic mean of precision and recall
        - weighted_accuracy: Accuracy weighted by trigger intensities (if provided)
    """
    true_positives = set(identified_triggers) & set(known_triggers)
    
    precision = len(true_positives) / len(identified_triggers) if identified_triggers else 0
    recall = len(true_positives) / len(known_triggers) if known_triggers else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    result = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    if trigger_intensities:
        # Calculate weighted accuracy based on trigger intensities
        weights = np.array([trigger_intensities.get(t, 1.0) for t in true_positives])
        result['weighted_accuracy'] = np.mean(weights) if weights.size > 0 else 0
    
    return result

def intervention_efficacy(
    pre_intervention: Dict[str, List[float]],
    post_intervention: Dict[str, List[float]]
) -> Dict[str, float]:
    """Measure the efficacy of interventions on clinical outcomes.
    
    Args:
        pre_intervention: Dictionary of metrics before intervention
        post_intervention: Dictionary of metrics after intervention
        
    Returns:
        Dictionary of intervention efficacy metrics
    """
    results = {}
    
    # Expected metrics
    metrics = ["pain_intensity", "duration_hours", "medication_doses", "functional_impact"]
    
    for metric in metrics:
        pre_values = pre_intervention.get(metric, [])
        post_values = post_intervention.get(metric, [])
        
        if not pre_values or not post_values:
            results[f"{metric}_change"] = 0.0
            results[f"{metric}_p_value"] = 1.0
            continue
        
        # Calculate mean change
        pre_mean = sum(pre_values) / len(pre_values)
        post_mean = sum(post_values) / len(post_values)
        percent_change = (post_mean - pre_mean) / pre_mean if pre_mean != 0 else 0.0
        
        # Calculate statistical significance
        t_stat, p_value = stats.ttest_ind(pre_values, post_values)
        
        results[f"{metric}_change"] = percent_change
        results[f"{metric}_p_value"] = p_value
    
    # Calculate overall efficacy score (weighted average of improvements)
    overall_efficacy = 0.0
    weights = {
        "pain_intensity_change": -0.4,  # Negative change is good
        "duration_hours_change": -0.3,  # Negative change is good
        "medication_doses_change": -0.2,  # Negative change is good
        "functional_impact_change": -0.1  # Negative change is good
    }
    
    for metric, weight in weights.items():
        change = results.get(metric, 0.0)
        # Adjust sign based on weight (negative changes are improvements)
        overall_efficacy += -1 * change * weight
    
    # Normalize to 0-1 scale
    overall_efficacy = max(0.0, min(1.0, overall_efficacy))
    results["overall_efficacy"] = overall_efficacy
    
    return results

def symptom_tracking_accuracy(
    predicted_symptoms: Dict[str, Any],
    actual_symptoms: Dict[str, Any],
    symptom_weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """Measure the accuracy of symptom tracking and prediction.
    
    Args:
        predicted_symptoms: Dictionary of predicted symptoms and intensities
        actual_symptoms: Dictionary of actual reported symptoms and intensities
        symptom_weights: Optional weights for different symptoms (default: equal weights)
        
    Returns:
        Dictionary of clinical accuracy metrics
    """
    if not predicted_symptoms or not actual_symptoms:
        return {
            "accuracy": 0.0,
            "weighted_accuracy": 0.0,
            "correlation": 0.0,
            "coverage": 0.0
        }
    
    # Default weights
    if symptom_weights is None:
        symptom_weights = {s: 1.0 for s in actual_symptoms.keys()}
    
    # Calculate metrics
    all_symptoms = set(actual_symptoms.keys()) | set(predicted_symptoms.keys())
    correct_symptoms = 0
    weighted_correct = 0.0
    total_weight = sum(symptom_weights.get(s, 1.0) for s in all_symptoms)
    
    # Prepare arrays for correlation
    actual_values = []
    predicted_values = []
    
    for symptom in all_symptoms:
        actual_value = actual_symptoms.get(symptom, 0)
        predicted_value = predicted_symptoms.get(symptom, 0)
        
        # For binary accuracy
        is_correct = (actual_value > 0 and predicted_value > 0) or (actual_value == 0 and predicted_value == 0)
        if is_correct:
            correct_symptoms += 1
            weighted_correct += symptom_weights.get(symptom, 1.0)
        
        # For correlation
        if symptom in actual_symptoms and symptom in predicted_symptoms:
            actual_values.append(actual_value)
            predicted_values.append(predicted_value)
    
    # Calculate correlation if we have enough data points
    correlation = 0.0
    if len(actual_values) >= 2:
        correlation, _ = stats.pearsonr(actual_values, predicted_values)
        correlation = max(0.0, correlation)  # Ensure non-negative
    
    # Calculate coverage (how many symptoms were predicted)
    coverage = len(predicted_symptoms) / len(actual_symptoms) if actual_symptoms else 0.0
    coverage = min(1.0, coverage)  # Cap at 1.0
    
    return {
        "accuracy": correct_symptoms / len(all_symptoms) if all_symptoms else 0.0,
        "weighted_accuracy": weighted_correct / total_weight if total_weight > 0 else 0.0,
        "correlation": correlation,
        "coverage": coverage
    }

def patient_reported_utility(
    survey_responses: List[Dict[str, Any]],
    min_response_rate: float = 0.7
) -> Dict[str, float]:
    """Analyze patient-reported utility metrics from survey data.
    
    Args:
        survey_responses: List of survey response dictionaries
        min_response_rate: Minimum required response rate for valid calculation
        
    Returns:
        Dictionary of patient-reported utility metrics
    """
    if not survey_responses:
        return {
            "overall_satisfaction": 0.0,
            "clinical_utility": 0.0,
            "usability": 0.0,
            "trust": 0.0,
            "response_rate": 0.0
        }
    
    # Expected survey fields (scale 0-10)
    fields = ["satisfaction", "clinical_utility", "usability", "trust"]
    
    # Calculate metrics
    metrics = {}
    for field in fields:
        values = [r.get(field, 0) for r in survey_responses if field in r]
        response_rate = len(values) / len(survey_responses)
        
        if response_rate >= min_response_rate and values:
            metrics[field] = sum(values) / len(values) / 10.0  # Normalize to 0-1
        else:
            metrics[field] = 0.0
    
    # Calculate overall satisfaction (weighted average)
    overall_satisfaction = (
        metrics.get("satisfaction", 0.0) * 0.4 +
        metrics.get("clinical_utility", 0.0) * 0.3 +
        metrics.get("usability", 0.0) * 0.2 +
        metrics.get("trust", 0.0) * 0.1
    )
    
    # Calculate average response rate
    avg_response_rate = sum(
        len([r for r in survey_responses if field in r]) / len(survey_responses)
        for field in fields
    ) / len(fields)
    
    return {
        "overall_satisfaction": overall_satisfaction,
        "clinical_utility": metrics.get("clinical_utility", 0.0),
        "usability": metrics.get("usability", 0.0),
        "trust": metrics.get("trust", 0.0),
        "response_rate": avg_response_rate
    }

def clinical_correlation(
    predicted_risk: List[float],
    actual_episodes: List[int],
    timeframe_hours: int = 24
) -> Dict[str, float]:
    """Measure correlation between predicted risk and actual migraine episodes.
    
    Args:
        predicted_risk: List of predicted risk scores (0-1)
        actual_episodes: Binary indicators of actual episodes (0 or 1)
        timeframe_hours: Prediction timeframe in hours
        
    Returns:
        Dictionary of clinical correlation metrics
    """
    if not predicted_risk or not actual_episodes or len(predicted_risk) != len(actual_episodes):
        return {
            "correlation": 0.0,
            "lead_time": 0.0,
            "false_alarm_rate": 0.0,
            "missed_episode_rate": 0.0
        }
    
    # Calculate correlation
    correlation, _ = stats.pearsonr(predicted_risk, actual_episodes)
    correlation = max(0.0, correlation)  # Ensure non-negative
    
    # Calculate false alarm rate (high risk but no episode)
    high_risk_threshold = 0.7
    high_risk_periods = [p >= high_risk_threshold for p in predicted_risk]
    false_alarms = sum(1 for hr, ae in zip(high_risk_periods, actual_episodes) if hr and not ae)
    false_alarm_rate = false_alarms / sum(high_risk_periods) if sum(high_risk_periods) > 0 else 0.0
    
    # Calculate missed episode rate (episode but low risk)
    low_risk_threshold = 0.3
    low_risk_periods = [p < low_risk_threshold for p in predicted_risk]
    missed_episodes = sum(1 for lr, ae in zip(low_risk_periods, actual_episodes) if lr and ae)
    missed_episode_rate = missed_episodes / sum(actual_episodes) if sum(actual_episodes) > 0 else 0.0
    
    # Estimate average lead time (simplified)
    lead_time = timeframe_hours * correlation
    
    return {
        "correlation": correlation,
        "lead_time": lead_time,
        "false_alarm_rate": false_alarm_rate,
        "missed_episode_rate": missed_episode_rate
    } 